from odinson.ruleutils.queryast import *

special_nodes = [
    # FieldConstraint 
    "Start-FieldConstraint-word",
    "End-FieldConstraint-word",
    "Start-FieldConstraint-lemma",
    "End-FieldConstraint-lemma",
    "Start-FieldConstraint-tag",
    "End-FieldConstraint-tag",

    # Constraints
    "Start-NotConstraint",
    "End-NotConstraint",
    "Start-AndConstraint",
    "End-AndConstraint",
    "Start-OrConstraint",
    "End-OrConstraint",

    # Surfaces
    "Start-TokenSurface",
    "End-TokenSurface",
    "Start-MentionSurface",
    "End-MentionSurface",
    "Start-ConcatSurface",
    "End-ConcatSurface",
    "Start-OrSurface",
    "End-OrSurface",
    "Start-RepeatSurface-?",
    "End-RepeatSurface-?",
    "Start-RepeatSurface-*",
    "End-RepeatSurface-*",
    "Start-RepeatSurface-+",
    "End-RepeatSurface-+",
]

# Linearize the tree
def linearize(ast_node: AstNode):
    node_type = type(ast_node).__name__
    if isinstance(ast_node, FieldConstraint):
        return [f'Start-FieldConstraint-{ast_node.name.string}', ast_node.value.string, f'End-FieldConstraint-{ast_node.name.string}']
    elif isinstance(ast_node, RepeatSurface):
        if ast_node.min == 0 and ast_node.max == 1:
            suffix = "?"
        elif ast_node.min == 0 and ast_node.max == None:
            suffix = "*"
        else:
            suffix = "+"
        return [f'Start-RepeatSurface-{suffix}'] + [y for x in ast_node.children() for y in linearize(x)] + [f'End-RepeatSurface-{suffix}']
    else:
        return [f'Start-{node_type}'] + [y for x in ast_node.children() for y in linearize(x)] + [f'End-{node_type}']
    