from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_surface

"""
Apply lambda on the AST
Filter the nodes based on @see condition
If a node satisfies @see condition, then we will apply @see operation on it
No recursive call after the application of the operation
"""
def apply_lambda_on_ast(node: AstNode, condition, operation) -> AstNode:
    if isinstance(node, FieldConstraint):
        if condition(node):
            return operation(node)
        else:
            return node
    elif isinstance(node, NotConstraint):
        if condition(node):
            return operation(node)
        else:
            return NotConstraint(apply_lambda_on_ast(node.constraint, condition, operation))
    elif isinstance(node, AndConstraint):
        if condition(node):
            return operation(node)
        else:
            return AndConstraint(apply_lambda_on_ast(node.lhs, condition, operation), apply_lambda_on_ast(node.rhs, condition, operation))
    elif isinstance(node, OrConstraint):
        if condition(node):
            return operation(node)
        else:
            return OrConstraint(apply_lambda_on_ast(node.lhs, condition, operation), apply_lambda_on_ast(node.rhs, condition, operation))
    elif isinstance(node, TokenSurface):
        if condition(node):
            return operation(node)
        else:
            return TokenSurface(apply_lambda_on_ast(node.constraint, condition, operation))
    elif isinstance(node, ConcatSurface):
        if condition(node):
            return operation(node)
        else:
            return ConcatSurface(apply_lambda_on_ast(node.lhs, condition, operation), apply_lambda_on_ast(node.rhs, condition, operation))
    elif isinstance(node, OrSurface):
        if condition(node):
            return operation(node)
        else:
            return OrSurface(apply_lambda_on_ast(node.lhs, condition, operation), apply_lambda_on_ast(node.rhs, condition, operation))
    elif isinstance(node, RepeatSurface):
        if condition(node):
            return operation(node)
        else:
            return RepeatSurface(apply_lambda_on_ast(node.c, condition, operation), node.min, node.max)
    else:
        raise ValueError(f"Unknown node: {node}")


if __name__ == "__main__":
    rule = parse_surface("[!word=this] [word=is] [word=a] [word=test]")
    print(
        apply_lambda_on_ast(
            rule, 
            lambda x: isinstance(x, FieldConstraint), 
            lambda x: FieldConstraint(x.name, ExactMatcher(s = 'that')) if x.name.string == 'word' and x.value.string == 'this' else x
            )
        )
    print(
        apply_lambda_on_ast(
            rule, 
            lambda x: isinstance(x, NotConstraint), 
            lambda x: x.constraint
            )
        )