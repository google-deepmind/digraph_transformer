# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parses python *function* into an AST with data and control flow edges.

This file is intentionally left to be as close as possible to `python-graphs`.
"""
from typing import Any, Dict, List, Set, Tuple, Union
import uuid

from absl import logging
import astunparse
import gast as ast
import numpy as np
from python_graphs import control_flow
from python_graphs import data_flow
from python_graphs import instruction as instruction_module
from python_graphs import program_graph as pg
from python_graphs import program_graph_dataclasses as pb
from six.moves import builtins

NEWLINE_TOKEN = '#NEWLINE#'
UNINDENT_TOKEN = '#UNINDENT#'
INDENT_TOKEN = '#INDENT#'

# Using the same names as OGBN-Code2
MASK_TOKEN = '_mask_'
NONE_TOKEN = '__NONE__'
UNKNOWN_TOKEN = '__UNK__'

# Aligned with OGB. Perhaps not all needed, but we keep them to avoid edge cases
FEATURE_FIELD_ORDER = ('name', 'arg', 'value', 's', 'n', 'id', 'attr')

# Some operations with commutative properties
# Exclude And as well as Or if you want to detect exceptions (e.g. and does
# not resolve subsequent argument if resolves to false)
COMMUTATIVE_OPERATORS = ('And', 'Or', 'Add', 'Mult', 'BitOr', 'BitXor',
                         'BitAnd')
COMMUTATIVE_EDGE_TYPES = ('body', 'finalbody', 'orelse')

FUNC_DEf = (ast.FunctionDef, ast.AsyncFunctionDef)


def py2ogbgraph(program: str,
                attr2idx: Dict[str, int],
                type2idx: Dict[str, int],
                mask_function_name: bool = True,
                max_characters_token: int = 100) -> Tuple[Dict[str, Any], str]:
  """The main function that converts a function into a graph.

  This is done similarly to the OGB Code2 dataset with the notable exception
  that nodes are not connected sequentially. Specifically, we construct the
  graph in a data-centric manner to achieve certain properties.

  Args:
    program: The function as string.
    attr2idx: maps attribute values to ids (OGB default).
    type2idx: maps attribute types to ids (OGB default).
    mask_function_name: If True, we mask out the function name (including
      recursive calls).
    max_characters_token: Limit on the maximum number of characters for values.

  Returns:
    A dictionary that (Imostly) contains np.ndarrays with the values required
    to construct the graph. This data structure is intentionally chosen to be
    as close as possible to the OGB data.
  """
  program_node = ast.parse(program, mode='exec')
  graph = get_program_graph(program_node)

  ast_nodes = list(ast.walk(program_node))

  assert all([not edge.has_back_edge for edge in graph.edges])
  assert len(ast_nodes) == len(graph.nodes)

  # Mask out target function name
  if mask_function_name:
    function_name = ast_nodes[1].name

    assert isinstance(ast_nodes[1], ast.FunctionDef) or isinstance(
        ast_nodes[1], ast.AsyncFunctionDef), (
            'To mask method name, 1st node in AST must be of type FunctionDef')
    node = graph.get_node_by_ast_node(ast_nodes[1])
    assert hasattr(node, 'fields')
    assert 'name' in node.fields
    assert node.fields['name'] == function_name

    ast_nodes[1].name = node.fields['name'] = MASK_TOKEN

    # Check for recursive calls
    for ast_node in ast_nodes:
      if isinstance(ast_node, ast.Call) and isinstance(ast_node.func, ast.Name):
        func_defs = list(graph.get_nodes_by_function_name(MASK_TOKEN))
        if not func_defs:
          continue

        ast_node.func.id = MASK_TOKEN
        for child in graph.children(graph.get_node_by_ast_node(ast_node)):
          if isinstance(child, ast.Name) and child.ast_node.id == function_name:
            child.ast_node.id = MASK_TOKEN
            if 'name' in child.fields and child.fields['name'] == function_name:
              child.fields['name'] = MASK_TOKEN

  ogb_walker = OGB_ASTWalker()
  ogb_walker.visit(program_node)

  graph2dataset_id = dict()
  dataset2graph_id = dict()
  for id_dataset, attributed_node in enumerate(ogb_walker.nodes):
    id_graph = graph.get_node_by_ast_node(attributed_node['node'])
    assert id_graph not in graph2dataset_id, f'Found id_graph={id_graph} twice'
    assert id_graph not in dataset2graph_id, f'Found id_dataset={id_dataset} twice'
    graph2dataset_id[id_graph.id] = id_dataset
    dataset2graph_id[id_dataset] = id_graph.id

  edge_index = []
  edges_deduplicated = list({(edge.id1, edge.id2, edge.field_name,
                              edge.type.value) for edge in graph.edges})
  for id1, id2, _, _ in edges_deduplicated:
    edge_index.append((graph2dataset_id[id1], graph2dataset_id[id2]))
  edge_index = np.array(edge_index).transpose()
  edge_coalesced_order = np.lexsort(np.flip(edge_index, axis=0))
  edge_index = edge_index[:, edge_coalesced_order]

  edge_type = []
  # Similarly to the labels, the encodings need to be handled at runtime
  edge_name = []
  edge_order = []
  for edge_idx in edge_coalesced_order:
    id1, id2, field_name, type_ = edges_deduplicated[edge_idx]
    order = 0
    if field_name is None:
      field_name = NONE_TOKEN
    elif ':' in field_name:
      splitted_name = field_name.split(':')
      field_name = ':'.join(splitted_name[:-1])
      if field_name not in COMMUTATIVE_EDGE_TYPES:
        order = int(splitted_name[-1])
    elif field_name == 'left':
      field_name = 'inputs'
    elif field_name == 'right':
      field_name = 'inputs'
      if not any([
          c for c in graph.children(graph.get_node_by_id(id1))
          if c.ast_type in COMMUTATIVE_OPERATORS
      ]):
        order = 1
    edge_type.append(type_)
    edge_name.append(field_name.encode('utf-8'))
    edge_order.append(order)

  node_feat_raw = []
  node_feat = []
  dfs_order = []
  depth = []
  attributed = []
  for idx, attributed_node in enumerate(ogb_walker.nodes):
    ast_node = attributed_node['node']

    node_type = attributed_node['type']

    fields = graph.get_node_by_ast_node(ast_node).fields
    for field in FEATURE_FIELD_ORDER:
      if field in fields:
        attribute = fields[field]
        break
    else:
      if fields.values():
        attribute = list(fields.values())[0]
      else:
        attribute = None
    is_attributed = attribute is not None
    if is_attributed:
      node_attr = str(attribute)[:max_characters_token]
      node_attr = node_attr.replace('\n', '').replace('\r', '')
    else:
      node_attr = NONE_TOKEN

    node_feat_raw.append(
        (str(node_type).encode('utf-8'), str(node_attr).encode('utf-8')))
    node_feat.append((type2idx.get(node_type,
                                   len(type2idx) - 1),
                      attr2idx.get(node_attr,
                                   len(attr2idx) - 1)))
    dfs_order.append(idx)
    depth.append(attributed_node['depth'])
    attributed.append(is_attributed)

  data = dict()
  # Nodes
  data['node_feat_raw'] = np.array(node_feat_raw, dtype='object')
  data['node_feat'] = np.array(node_feat, dtype=np.int64)
  data['node_dfs_order'] = np.array(dfs_order, dtype=np.int64).reshape(-1, 1)
  data['node_depth'] = np.array(depth, dtype=np.int64).reshape(-1, 1)
  data['node_is_attributed'] = np.array(
      attributed, dtype=np.int64).reshape(-1, 1)

  # Edges
  data['edge_index'] = edge_index
  data['edge_type'] = np.array(edge_type, dtype=np.int64).reshape(-1, 1)
  data['edge_name'] = np.array(edge_name, dtype='object').reshape(-1, 1)
  data['edge_order'] = np.array(edge_order, dtype=np.int64).reshape(-1, 1)

  # Sizes
  data['num_nodes'] = len(data['node_feat'])
  data['num_edges'] = len(data['edge_index'][0])

  return data, function_name


class OGB_ASTWalker(ast.NodeVisitor):  # pylint: disable=invalid-name
  """Minimal version of the Open Graph Benchmark ASTWalker."""

  def __init__(self):
    self.node_id = 0
    self.stack = []
    self.nodes = []

  def generic_visit(self, node: ast.Node):
    # encapsulate all node features in a dict
    self.nodes.append({
        'type': type(node).__name__,
        'node': node,
        'depth': len(self.stack)
    })

    # DFS traversal logic
    self.stack.append(node)
    super().generic_visit(node)
    self.stack.pop()


def get_program_graph(program: Union[str, ast.AST]):
  """Constructs a program graph to represent the given program."""
  if isinstance(program, ast.AST):
    program_node = program
  else:
    program_node = ast.parse(program, mode='exec')

  program_graph = pg.ProgramGraph()

  # Perform control flow analysis.
  control_flow_graph = control_flow.get_control_flow_graph(program_node)

  # Add AST_NODE program graph nodes corresponding to Instructions in the
  # control flow graph.
  for control_flow_node in control_flow_graph.get_control_flow_nodes():
    program_graph.add_node_from_instruction(control_flow_node.instruction)

  # Add AST_NODE program graph nodes corresponding to AST nodes.
  for ast_node in ast.walk(program_node):
    if not program_graph.contains_ast_node(ast_node):
      pg_node = pg.make_node_from_ast_node(ast_node)
      program_graph.add_node(pg_node)

  root = program_graph.get_node_by_ast_node(program_node)
  program_graph.root_id = root.id

  # Add AST edges (FIELD). Also add AST_LIST and AST_VALUE program graph nodes.
  for ast_node in ast.walk(program_node):
    node = program_graph.get_node_by_ast_node(ast_node)
    setattr(node, 'fields', {})
    for field_name, value in list(ast.iter_fields(ast_node)):
      if isinstance(value, list):
        last_item = None
        for index, item in enumerate(value):
          list_field_name = make_list_field_name(field_name, index)
          if isinstance(item, ast.AST):
            if last_item is not None:
              assert isinstance(item, ast.AST)
            program_graph.add_new_edge(ast_node, item, pb.EdgeType.FIELD,
                                       list_field_name)
          else:
            if last_item is not None:
              assert not isinstance(item, ast.AST)
            node.fields[list_field_name] = item
          last_item = item
      elif isinstance(value, ast.AST):
        program_graph.add_new_edge(ast_node, value, pb.EdgeType.FIELD,
                                   field_name)
      else:
        node.fields[field_name] = value

  # Perform data flow analysis.
  analysis = data_flow.LastAccessAnalysis()
  for node in control_flow_graph.get_enter_control_flow_nodes():
    analysis.visit(node)

  # Add control flow edges (NEXT_SYNTAX) - as orginially done by python graphs
  # for CFG_NEXT.
  for control_flow_node in control_flow_graph.get_control_flow_nodes():
    instruction = control_flow_node.instruction
    for next_control_flow_node in control_flow_node.next:
      next_instruction = next_control_flow_node.instruction
      program_graph.add_new_edge(
          instruction.node,
          next_instruction.node,
          edge_type=pb.EdgeType.NEXT_SYNTAX)
      # edge_type=pb.EdgeType.CFG_NEXT)

  # Add data flow edges (LAST_READ and LAST_WRITE).
  for control_flow_node in control_flow_graph.get_control_flow_nodes():
    # Start with the most recent accesses before this instruction.
    last_accesses = control_flow_node.get_label('last_access_in').copy()
    for access in control_flow_node.instruction.accesses:
      # Extract the node and identifiers for the current access.
      pg_node = program_graph.get_node_by_access(access)
      access_name = instruction_module.access_name(access)
      write_identifier = instruction_module.access_identifier(
          access_name, 'write')
      for write in last_accesses.get(write_identifier, []):
        write_pg_node = program_graph.get_node_by_access(write)
        program_graph.add_new_edge(
            write_pg_node, pg_node, edge_type=pb.EdgeType.LAST_WRITE)
      # Update the state to refer to this access as the most recent one.
      if instruction_module.access_is_write(access):
        last_accesses[write_identifier] = [access]

  # Add COMPUTED_FROM edges.
  for node in ast.walk(program_node):
    if isinstance(node, ast.Assign):
      for value_node in ast.walk(node.value):
        if isinstance(value_node, ast.Name):
          # TODO(dbieber): If possible, improve precision of these edges.
          for target in node.targets:
            program_graph.add_new_edge(
                value_node, target, edge_type=pb.EdgeType.COMPUTED_FROM)

  # Add CALLS, FORMAL_ARG_NAME and RETURNS_TO edges.
  for node in ast.walk(program_node):
    if isinstance(node, ast.Call):
      if isinstance(node.func, ast.Name):
        # TODO(dbieber): Use data flow analysis instead of all function defs.
        func_defs = list(program_graph.get_nodes_by_function_name(node.func.id))
        # For any possible last writes that are a function definition, add the
        # formal_arg_name and returns_to edges.
        if not func_defs:
          # TODO(dbieber): Add support for additional classes of functions,
          # such as attributes of known objects and builtins.
          if node.func.id in dir(builtins):
            message = 'Function is builtin.'
          else:
            message = 'Cannot statically determine the function being called.'
          logging.debug('%s (%s)', message, node.func.id)
        for func_def in func_defs:
          fn_node = func_def.node
          # Add calls edge from the call node to the function definition.
          program_graph.add_new_edge(node, fn_node, edge_type=pb.EdgeType.CALLS)
          # Add returns_to edges from the function's return statements to the
          # call node.
          for inner_node in ast.walk(func_def.node):
            # TODO(dbieber): Determine if the returns_to should instead go to
            # the next instruction after the Call node instead.
            if isinstance(inner_node, ast.Return):
              program_graph.add_new_edge(
                  inner_node, node, edge_type=pb.EdgeType.RETURNS_TO)

          # Add formal_arg_name edges from the args of the Call node to the
          # args in the FunctionDef.
          for index, arg in enumerate(node.args):
            formal_arg = None
            if index < len(fn_node.args.args):
              formal_arg = fn_node.args.args[index]
            elif fn_node.args.vararg:
              # Since args.vararg is a string, we use the arguments node.
              # TODO(dbieber): Use a node specifically for the vararg.
              formal_arg = fn_node.args
            if formal_arg is not None:
              # Note: formal_arg can be an AST node or a string.
              program_graph.add_new_edge(
                  arg, formal_arg, edge_type=pb.EdgeType.FORMAL_ARG_NAME)
            else:
              # TODO(dbieber): If formal_arg is None, then remove all
              # formal_arg_name edges for this FunctionDef.
              logging.debug('formal_arg is None')
          for keyword in node.keywords:
            name = keyword.arg
            formal_arg = None
            for arg in fn_node.args.args:
              if isinstance(arg, ast.Name) and arg.id == name:
                formal_arg = arg
                break
            else:
              if fn_node.args.kwarg:
                # Since args.kwarg is a string, we use the arguments node.
                # TODO(dbieber): Use a node specifically for the kwarg.
                formal_arg = fn_node.args
            if formal_arg is not None:
              program_graph.add_new_edge(
                  keyword.value,
                  formal_arg,
                  edge_type=pb.EdgeType.FORMAL_ARG_NAME)
            else:
              # TODO(dbieber): If formal_arg is None, then remove all
              # formal_arg_name edges for this FunctionDef.
              logging.debug('formal_arg is None')
      else:
        # TODO(dbieber): Add a special case for Attributes.
        logging.debug(
            'Cannot statically determine the function being called. (%s)',
            astunparse.unparse(node.func).strip())

  refined_cf_visitor = ControlFlowVisitor(program_graph, control_flow_graph)
  refined_cf_visitor.run(root)

  for control_flow_node in refined_cf_visitor.graph.get_control_flow_nodes():
    instruction = control_flow_node.instruction
    for next_control_flow_node in control_flow_node.next:
      next_instruction = next_control_flow_node.instruction
      program_graph.add_new_edge(
          instruction.node,
          next_instruction.node,
          edge_type=pb.EdgeType.CFG_NEXT)

  return program_graph


class CustomControlFlowWalker(ast.NodeVisitor):
  """This additional control flow walker, analyzes the possible orders in which the instructions can be executed.
  """

  def __init__(self, program_graph: pg.ProgramGraph,
               cfg_graph: control_flow.ControlFlowGraph):
    self.program_graph = program_graph
    self.cfg_graph = cfg_graph
    self.children_order: Dict[int, List[int]] = dict()
    self.node_with_id_visited = set()

  def return_exceptional_cf_node(self,
                                 node: pg.ProgramGraphNode) -> Set[ast.AST]:
    if (type(node.ast_node) in [
        ast.Assert, ast.Break, ast.Raise, ast.Continue, ast.Return, ast.Yield
    ]):
      return {node.ast_node}
    else:
      return set()

  def prune_exceptional_cf_nodes(
      self, node: pg.ProgramGraphNode,
      exceptional_nodes: Set[ast.AST]) -> Set[ast.AST]:
    if type(node.ast_node) in [ast.For, ast.While]:
      return exceptional_nodes - {ast.Break, ast.Continue}
    elif isinstance(node.ast_node, ast.If):
      return exceptional_nodes - {ast.Assert, ast.Raise}
    else:
      return exceptional_nodes

  def generic_visit(self, node) -> Tuple[Set[int], Set[int], Set[int]]:
    neighbors = self.program_graph.neighbors_map[node.id]

    children = []
    for edge, _ in neighbors:
      # Look at outgoing ast edges for the children
      if edge.id1 != node.id or edge.type != pb.EdgeType.FIELD:
        continue
      child = self.program_graph.get_node_by_id(edge.id2)
      descendants, depends_on, exceptional_nodes = super().visit(child)
      children.append(
          (edge.id2, edge, descendants, depends_on, exceptional_nodes))

    node_depends_on = set()
    for edge, _ in neighbors:
      # Look at incoming data flow dependencies
      if (edge.id2 == node.id and
          edge.type in [pb.EdgeType.LAST_WRITE, pb.EdgeType.COMPUTED_FROM] and
          # Only allow edge dependence in sequential order
          edge.id1 in self.node_with_id_visited):
        node_depends_on.add(edge.id1)

    self.node_with_id_visited.add(node.id)

    if not children:
      return {node.id}, node_depends_on, self.return_exceptional_cf_node(node)

    blocks = {
        field:
        [child for child in children if child[1].field_name.startswith(field)]
        for field in COMMUTATIVE_EDGE_TYPES
    }
    self.children_order[node.id] = {}

    for field, block in blocks.items():
      current_parent = -1
      predecessor_list = list()
      successor_list = list()

      if block:
        entry_node_id = block[0][0]
      else:
        entry_node_id = None

      # Required to determine if exceptional node overrules dataflow
      children_order = []
      exceptional_node_order = 0

      for idx, (_, _, descendants, depends_on,
                exceptional_nodes) in enumerate(block):
        children_order.append(exceptional_node_order)
        predecessor_list.append(set())
        successor_list.append(set())

        if not depends_on and not exceptional_nodes:
          if current_parent >= 0:
            predecessor_list[idx].add(current_parent)
            successor_list[current_parent].add(idx)
          continue

        for previous_child_idx in reversed(range(idx)):
          if children_order[previous_child_idx] < exceptional_node_order:
            break

          for dependence in depends_on:
            if dependence in block[previous_child_idx][2]:
              predecessor_list[idx].add(previous_child_idx)
              successor_list[previous_child_idx].add(idx)
              children_order[idx] = max(children_order[-1],
                                        children_order[previous_child_idx] + 1)

        if exceptional_nodes:
          current_parent = idx
          for previous_child_idx in range(idx):
            if not successor_list[previous_child_idx]:
              successor_list[previous_child_idx].add(idx)
              predecessor_list[idx].add(previous_child_idx)

          exceptional_node_order = max(children_order) + 1
          children_order[idx] = exceptional_node_order
        elif not predecessor_list[idx] and current_parent >= 0:
          predecessor_list[idx].add(current_parent)
          successor_list[current_parent].add(idx)

      self.children_order[node.id][field] = [
          entry_node_id, predecessor_list, successor_list
      ]

    agg_descendants = set().union(
        *[descendants for _, _, descendants, _, _ in children])
    agg_descendants |= {node.id}

    agg_depends_on = set().union(
        *[depends_on for _, _, _, depends_on, _ in children])
    agg_depends_on |= node_depends_on
    agg_depends_on -= agg_descendants

    agg_exceptional_nodes = set().union(
        *[exceptional_nodes for _, _, _, _, exceptional_nodes in children])
    agg_exceptional_nodes |= self.return_exceptional_cf_node(node)
    agg_exceptional_nodes = self.prune_exceptional_cf_nodes(
        node, agg_exceptional_nodes)

    return agg_descendants, agg_depends_on, agg_exceptional_nodes


class ControlFlowVisitor(control_flow.ControlFlowVisitor):
  """Here we overwrite the way how `bodies` in the AST are ordered based on the insights in which order instructions can be executed.
  """

  def __init__(self, program_graph: pg.ProgramGraph,
               cfg_graph: control_flow.ControlFlowGraph):
    super().__init__()
    self.program_graph = program_graph
    self.refined_cf_walker = CustomControlFlowWalker(program_graph, cfg_graph)

  def run(self, root: pg.ProgramGraphNode):
    self.refined_cf_walker.visit(root)
    start_block = self.graph.start_block
    end_block = self.visit(root.ast_node, start_block)
    exit_block = self.new_block(
        node=root.ast_node, label='<exit>', prunable=False)
    end_block.add_exit(exit_block)
    self.graph.compact()

  def visit_list(self, items, current_block):
    """Visit each of the items in a list from the AST."""

    if len(items) < 2:
      for item in items:
        current_block = self.visit(item, current_block)
      return current_block

    parent = self.program_graph.parent(
        self.program_graph.get_node_by_ast_node(items[0]))
    children_order = self.refined_cf_walker.children_order[parent.id]
    children_order = [
        (field, co)
        for field, co in children_order.items()
        if co[0] == self.program_graph.get_node_by_ast_node(items[0]).id
    ][0]
    field, (_, predecessor_list, successor_list) = children_order

    assert not predecessor_list[0], 'First entrty cannot have predecessor'
    assert not successor_list[-1], 'Last entrty cannot have successor'

    entry_block = current_block

    item_idx_to_block = []
    for item_idx, predecessors in enumerate(predecessor_list):
      item = items[item_idx]

      current_block = self.new_block(
          node=item if predecessors else entry_block.node,  # For consistency
          label=f'field_{item_idx}')

      if not predecessors:
        entry_block.add_exit(current_block)
      else:
        for pred_idx in predecessors:
          item_idx_to_block[pred_idx].add_exit(current_block)

      current_block = self.visit(item, current_block)
      item_idx_to_block.append(current_block)

    after_block = self.new_block(node=entry_block.node, label='after_block')
    for item_idx, successor in enumerate(successor_list):
      if not successor:
        item_idx_to_block[item_idx].add_exit(after_block)

    return after_block


def make_list_field_name(field_name, index):
  return '{}:{}'.format(field_name, index)


def parse_list_field_name(list_field_name):
  field_name, index = list_field_name.split(':')
  index = int(index)
  return field_name, index


def unique_id():
  """Returns a unique id that is suitable for identifying graph nodes."""
  return uuid.uuid4().int & ((1 << 64) - 1)
