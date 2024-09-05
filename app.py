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

# Definição do Autômato Finito Determinístico (AFD)
class AFD:
    def __init__(self):
        self.estados = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5']
        self.estado_atual = 'q0'
        self.transicoes = {
            'q0': {'digito': 'q1', 'letra': 'q2', 'op': 'q3', 'espaco': 'q0', 'aspas': 'q4', 'comp': 'q5'},
            'q1': {'digito': 'q1', 'op': 'q3', 'espaco': 'q0'},  # Estado para números
            'q2': {'letra': 'q2', 'digito': 'q2', 'op': 'q3', 'espaco': 'q0', 'aspas': 'q4'},  # Estado para variáveis e identificadores
            'q3': {'letra': 'q2', 'digito': 'q1', 'op': 'q3', 'espaco': 'q0', 'aspas': 'q4'},  # Estado para operadores
            'q4': {'aspas': 'q0', 'caractere': 'q4'},  # Estado de leitura da string, aceita qualquer caractere exceto aspas de fechamento
            'q5': {'espaco': 'q0', 'op': 'q3', 'letra': 'q2', 'digito': 'q1', 'RPAREN': 'q0'}  # Estado para fechamento da string e inclusão de parênteses
        }
        self.estados_finais = ['q1', 'q2', 'q3', 'q4', 'q5']

    def transicao(self, caractere):
        """ Faz a transição do estado baseado no caractere de entrada. """
        if self.estado_atual == 'q4':
            tipo_caractere = 'caractere'  # Qualquer caractere dentro de uma string (exceto aspas de fechamento)
        else:
            if caractere.isdigit():
                tipo_caractere = 'digito'
            elif caractere.isalpha():
                tipo_caractere = 'letra'
            elif caractere in "+-*/=(){}:":
                tipo_caractere = 'op'
            elif caractere in "\"'":
                tipo_caractere = 'aspas'
            elif caractere in "<>!=":
                tipo_caractere = 'comp'
            elif caractere in " \t\n":
                tipo_caractere = 'espaco'
            else:
                tipo_caractere = 'caractere'

        # Verificação da transição
        if self.estado_atual in self.transicoes and tipo_caractere in self.transicoes[self.estado_atual]:
            self.estado_atual = self.transicoes[self.estado_atual][tipo_caractere]
        else:
            st.error(f"Erro: Não há transição válida para o estado {self.estado_atual} com o caractere '{caractere}'.")
            self.estado_atual = 'q0'  # Reiniciar o estado
        return self.estado_atual

    def reset(self):
        """ Reinicia o estado do AFD. """
        self.estado_atual = 'q0'

# Função para desenhar o autômato finito determinístico (AFD)
def plotar_afd(afd):
    grafo_afd = nx.DiGraph()

    # Adicionar nós e transições
    for estado, transicoes in afd.transicoes.items():
        grafo_afd.add_node(estado)
        for entrada, proximo_estado in transicoes.items():
            grafo_afd.add_edge(estado, proximo_estado, label=entrada)

    pos = nx.spring_layout(grafo_afd)

    # Tamanho maior para a visualização
    plt.figure(figsize=(10, 7))

    # Destacar estado inicial e finais
    initial_state = 'q0'
    final_states = afd.estados_finais

    # Desenhar todos os estados
    node_colors = []
    for node in grafo_afd.nodes:
        if node == initial_state:
            node_colors.append('green')  # Cor verde para o estado inicial
        elif node in final_states:
            node_colors.append('red')  # Cor vermelha para os estados finais
        else:
            node_colors.append('lightblue')  # Cor padrão para os outros estados

    nx.draw(grafo_afd, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=12, font_weight="bold", arrows=True)

    # Desenhar as arestas com rótulos
    labels = nx.get_edge_attributes(grafo_afd, 'label')
    nx.draw_networkx_edge_labels(grafo_afd, pos, edge_labels=labels)

    # Exibir a figura no Streamlit
    st.subheader("Autômato Finito Determinístico (DFA) - Estado Inicial e Estados Finais Destacados")
    st.pyplot(plt)


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
                    defined_vars.add(stmt[1])
                elif stmt[0] in ['print', 'if', 'if_else', 'while']:
                    for expr in traverse_expression(stmt[1]):
                        if expr not in defined_vars and expr not in variables:
                            errors.append(f"Erro semântico: Variável '{expr}' utilizada antes de ser definida.")
                if stmt[0] == 'if_else':
                    traverse_statements(stmt[2])
                    traverse_statements(stmt[3])
                elif stmt[0] == 'while':
                    traverse_statements(stmt[2])

    def traverse_expression(expr):
        vars_in_expr = set()
        if isinstance(expr, tuple):
            if expr[0] in ['PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'LT', 'GT', 'LE', 'GE', 'EQ', 'NE']:
                vars_in_expr.update(traverse_expression(expr[1]))
                vars_in_expr.update(traverse_expression(expr[2]))
            elif expr[0] == 'NAME':
                vars_in_expr.add(expr[1])
        return vars_in_expr

# Função para construir a árvore sintática como um grafo hierárquico
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

def hierarchy_pos(G, root=None, width=1., vert_gap=0.5, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.5, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    if parsed is None:
        parsed = []

    children = list(G.neighbors(root))
    if len(children) != 0:
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    
    return pos

# Função para plotar a árvore sintática
def plot_tree(graph):
    # Ajustar o espaçamento dos nós na árvore para evitar sobreposição
    pos = hierarchy_pos(graph, 0, width=3., vert_gap=0.8)  # Aumentar o espaçamento

    labels = nx.get_node_attributes(graph, 'label')

    # Aumentar o tamanho da figura e ajustar o espaçamento dos nós
    plt.figure(figsize=(18, 12))  # Figura maior para melhor visualização

    # Adicionar nós e arestas
    nx.draw(graph, pos, labels=labels, with_labels=True, arrows=False, node_size=3000, node_color='lightblue', font_size=12)

    # Ajustar os rótulos para que fiquem visíveis e não se sobreponham
    for node, (x, y) in pos.items():
        plt.text(x, y, s=labels[node], bbox=dict(facecolor='white', alpha=0.6), horizontalalignment='center', fontsize=10)

    # Exibir a árvore no Streamlit
    st.subheader("Árvore Sintática")
    st.pyplot(plt)


# Função para traduzir a árvore sintática em código Assembly fictício
def translate_to_assembly(tokens):
    assembly_code = []
    label_counter = 0

    for token in tokens:
        if token.type == 'PRINT':
            assembly_code.append(f'PRINT {token.value}')
        elif token.type == 'EQUALS':
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
st.title("Simulador de Compilador Python para Assembly com AFD - CC6NA")
st.subheader("Professor: Fábio Araújo")
st.subheader("Monitores: João Renan E Carlos Egger")

# Input do código fonte
codigo_fonte = st.text_area("Digite um código em Python:")

# Instância do AFD
afd = AFD()

# Botão para iniciar a compilação/interpretação e o AFD
if st.button("Compilar/Interpretar com AFD"):
    if codigo_fonte:
        lexer.input(codigo_fonte)
        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append(tok)

        # Resetar o AFD para cada nova análise
        afd.reset()
        estados_afd = []
        for caractere in codigo_fonte:
            estado = afd.transicao(caractere)
            if estado:
                estados_afd.append(estado)

        # Exibir tokens analisados
        st.sidebar.write("Tokens:", [str(t) for t in tokens])

        # Exibir estados percorridos pelo AFD
        st.sidebar.write("Estados AFD percorridos:", estados_afd)

        # Visualizar o AFD
        st.sidebar.write("Autômato Finito Determinístico:")
        plotar_afd(afd)

        # Análise sintática
        resultado = parser.parse(codigo_fonte, lexer=lexer)

        # Construção da árvore sintática a partir do resultado da análise sintática
        arvore_sintatica = build_syntax_tree(resultado)

        # Plotagem da árvore sintática
        st.sidebar.write("Árvore Sintática:")
        plot_tree(arvore_sintatica)

        # Geração e exibição do código Assembly
        st.sidebar.write("Código Assembly:")
        codigo_assembly = translate_to_assembly(tokens)
        st.sidebar.text("\n".join(codigo_assembly))

        # Verificações semânticas
        erros_semanticos = check_semantics(tokens)
        if erros_semanticos:
            st.sidebar.write("Erros Semânticos:")
            for erro in erros_semanticos:
                st.sidebar.error(erro)
        else:
            st.sidebar.write("Nenhum erro semântico encontrado.")
    else:
        st.warning("Por favor, insira um código Python.")
