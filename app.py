import streamlit as st
from ply import lex, yacc
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Compiladores CC6NA", page_icon='🎃')

# Analisador Léxico para Python
tokens = (
    'NUMBER', 'STRING', 'NAME', 'EQUALS', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
    'LPAREN', 'RPAREN', 'PRINT', 'IF', 'ELSE', 'WHILE', 'COLON', 'LT', 'GT', 'LE', 'GE', 'EQ', 'NE',
)

t_EQUALS = r'='
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COLON = r':'
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_GE = r'>='
t_EQ = r'=='
t_NE = r'!='
t_ignore = ' \t'

def t_PRINT(t):
    r'print'
    return t

def t_IF(t):
    r'if'
    return t

def t_ELSE(t):
    r'else'
    return t

def t_WHILE(t):
    r'while'
    return t

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    return t

def t_STRING(t):
    r'\"([^\\\n]|(\\.))*?\"'
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    st.error(f"Erro léxico na linha {t.lineno}: Caracter inválido '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# Analisador Sintático para Python básico
def p_program(p):
    '''program : statement_list'''
    p[0] = p[1]

def p_statement_list(p):
    '''statement_list : statement
                      | statement_list statement'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_statement(p):
    '''statement : expression_statement
                 | print_statement
                 | assignment_statement
                 | if_statement
                 | while_statement'''
    p[0] = p[1]

def p_expression_statement(p):
    'expression_statement : expression'
    p[0] = p[1]

def p_print_statement(p):
    'print_statement : PRINT LPAREN expression RPAREN'
    p[0] = ('print', p[3])

def p_assignment_statement(p):
    'assignment_statement : NAME EQUALS expression'
    p[0] = ('assign', p[1], p[3])

def p_if_statement(p):
    '''if_statement : IF expression COLON statement_list else_statement'''
    if p[5] is None:
        p[0] = ('if', p[2], p[4])
    else:
        p[0] = ('if_else', p[2], p[4], p[5])

def p_else_statement(p):
    '''else_statement : ELSE COLON statement_list
                      | empty'''
    if len(p) == 4:
        p[0] = p[3]
    else:
        p[0] = None

def p_while_statement(p):
    'while_statement : WHILE expression COLON statement_list'
    p[0] = ('while', p[2], p[4])

def p_expression(p):
    '''expression : term
                  | expression PLUS term
                  | expression MINUS term
                  | expression LT term
                  | expression GT term
                  | expression LE term
                  | expression GE term
                  | expression EQ term
                  | expression NE term'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = (p[2], p[1], p[3])

def p_term(p):
    '''term : factor
            | term TIMES factor
            | term DIVIDE factor'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = (p[2], p[1], p[3])

def p_factor(p):
    '''factor : NUMBER
              | STRING
              | NAME
              | LPAREN expression RPAREN'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]

def p_empty(p):
    'empty :'
    p[0] = None

def p_error(p):
    if p:
        st.error(f"Erro sintático na linha {p.lineno}, no token '{p.value}'")
    else:
        st.error("Erro sintático no final da entrada")

parser = yacc.yacc()

# Função para realizar verificações semânticas simples

def check_semantics(statements):
    errors = []
    variables = set()
    defined_vars = set()

    def traverse_statements(stmts):
        for stmt in stmts:
            if isinstance(stmt, tuple):
                if stmt[0] == 'assign':
                    # Adiciona a variável à lista de definidas
                    defined_vars.add(stmt[1])
                elif stmt[0] in ['print', 'if', 'if_else', 'while']:
                    # Verifica se todas as variáveis usadas estão definidas
                    for expr in traverse_expression(stmt[1]):
                        if expr not in defined_vars and expr not in variables:
                            errors.append(f"Erro semântico: Variável '{expr}' utilizada antes de ser definida.")
                if stmt[0] == 'if_else':
                    traverse_statements(stmt[2])  # Parte 'then' do if
                    traverse_statements(stmt[3])  # Parte 'else' do if
                elif stmt[0] == 'while':
                    traverse_statements(stmt[2])  # Corpo do while

    def traverse_expression(expr):
        """ Extrai variáveis usadas em uma expressão """
        vars_in_expr = set()
        if isinstance(expr, tuple):
            if expr[0] in ['PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'LT', 'GT', 'LE', 'GE', 'EQ', 'NE']:
                vars_in_expr.update(traverse_expression(expr[1]))
                vars_in_expr.update(traverse_expression(expr[2]))
            elif expr[0] == 'NAME':
                vars_in_expr.add(expr[1])
        return vars_in_expr

# Função para construir a árvore sintática como um grafo hierárquico a partir dos tokens
def build_syntax_tree(statements):
    graph = nx.DiGraph()
    node_id = 0

    def add_node(label, parent=None):
        nonlocal node_id
        graph.add_node(node_id, label=label)
        if parent is not None:
            graph.add_edge(parent, node_id)
        node_id += 1
        return node_id - 1

    def process_statement(statement, parent=None):
        if isinstance(statement, tuple):
            node = add_node(statement[0], parent)
            for child in statement[1:]:
                process_statement(child, node)
        else:
            add_node(statement, parent)

    root = add_node("Program")
    for stmt in statements:
        process_statement(stmt, root)

    return graph

# Função para gerar a posição hierárquica dos nós
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    if parsed is None:
        parsed = []

    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph):
        raise TypeError('G must be a DiGraph')

    if len(children) != 0:
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    
    return pos

# Função para plotar a árvore sintática
def plot_tree(graph):
    pos = hierarchy_pos(graph, 0)  # 0 é o ID da raiz
    labels = nx.get_node_attributes(graph, 'label')
    plt.figure(figsize=(14, 10))  # Aumenta o tamanho da figura para melhor visualização
    nx.draw(graph, pos, labels=labels, with_labels=True, arrows=False, node_size=2000, node_color='lightblue', font_size=12)
    st.pyplot(plt)
# Função para traduzir a árvore sintática em código Assembly fictício
def translate_to_assembly(tokens):
    assembly_code = []
    label_counter = 0

    for token in tokens:
        if token.type == 'PRINT':
            assembly_code.append(f'PRINT {token.value}')
        elif token.type == 'EQUALS':
            # Assume que a instrução anterior contém o lado esquerdo e a próxima contém o lado direito
            lhs = tokens[tokens.index(token) - 1]
            rhs = tokens[tokens.index(token) + 1]
            assembly_code.append(f'MOV {lhs.value}, {rhs.value}')
        elif token.type in ['PLUS', 'MINUS', 'TIMES', 'DIVIDE']:
            operation = {
                'PLUS': 'ADD',
                'MINUS': 'SUB',
                'TIMES': 'MUL',
                'DIVIDE': 'DIV'
            }[token.type]
            lhs = tokens[tokens.index(token) - 1]
            rhs = tokens[tokens.index(token) + 1]
            assembly_code.append(f'{operation} {lhs.value}, {rhs.value}')
        elif token.type == 'IF':
            label_counter += 1
            condition = tokens[tokens.index(token) + 1]
            assembly_code.append(f'CMP {condition.value}')
            assembly_code.append(f'JLE ELSE_{label_counter}')
        elif token.type == 'ELSE':
            assembly_code.append(f'JMP ENDIF_{label_counter}')
            assembly_code.append(f'ELSE_{label_counter}:')
        elif token.type == 'WHILE':
            label_counter += 1
            condition = tokens[tokens.index(token) + 1]
            assembly_code.append(f'WHILE_{label_counter}:')
            assembly_code.append(f'CMP {condition.value}')
            assembly_code.append(f'JGE ENDWHILE_{label_counter}')
        elif token.type == 'RPAREN':
            assembly_code.append(f'ENDIF_{label_counter}:')
            label_counter += 1

    return assembly_code

# Interface do Streamlit
st.title("Simulador de Compilador Python para Assembly")

# Input do código fonte
codigo_fonte = st.text_area("Digite um código em Python:")

# Botão para iniciar a compilação/interpretação
if st.button("Compilar/Interpretar"):
    if codigo_fonte:
        lexer.input(codigo_fonte)
        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append(tok)

        st.sidebar.write("Tokens:", [str(t) for t in tokens])

        # Análise sintática
        result = parser.parse(codigo_fonte, lexer=lexer)

        # Construção da árvore sintática a partir do resultado da análise sintática
        syntax_tree = build_syntax_tree(result)

        # Plotagem da árvore sintática
        st.sidebar.write("Árvore Sintática:")
        plot_tree(syntax_tree)

        # Geração e exibição do código Assembly
        st.sidebar.write("Código Assembly:")
        assembly_code = translate_to_assembly(tokens)
        st.sidebar.text("\n".join(assembly_code))

        # Verificações semânticas
        semantic_errors = check_semantics(tokens)
        if semantic_errors:
            st.sidebar.write("Erros Semânticos:")
            for error in semantic_errors:
                st.sidebar.error(error)
        else:
            st.sidebar.write("Nenhum erro semântico encontrado.")
    else:
        st.warning("Por favor, insira um código Python.")






