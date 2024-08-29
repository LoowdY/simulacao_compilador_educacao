# Simulador de Compilador Python para Assembly

## Objetivos do Software

O "Simulador de Compilador Python para Assembly" é uma ferramenta educacional desenvolvida para auxiliar estudantes e entusiastas de Ciência da Computação a compreenderem o funcionamento básico de um compilador. Através de uma interface interativa, o software permite que os usuários escrevam código em Python, que é então analisado lexicamente, sintaticamente, e traduzido para um código Assembly fictício. O objetivo principal é proporcionar uma experiência prática no entendimento dos conceitos de análise léxica, análise sintática e geração de código, que são fundamentais na construção de compiladores.

## Funcionalidades

### 1. **Analisador Léxico**

O analisador léxico (lexer) é responsável por ler o código-fonte Python e dividir o texto em unidades menores chamadas de "tokens". Cada token representa um elemento básico da linguagem, como números, operadores, identificadores, palavras-chave (e.g., `if`, `else`, `while`), etc.

- **Tokens Reconhecidos:**
  - Números (`NUMBER`)
  - Strings (`STRING`)
  - Identificadores (`NAME`)
  - Operadores (`PLUS`, `MINUS`, `TIMES`, `DIVIDE`)
  - Comparadores (`LT`, `GT`, `LE`, `GE`, `EQ`, `NE`)
  - Estruturas de controle (`IF`, `ELSE`, `WHILE`)
  - Entre outros.

### 2. **Analisador Sintático**

O analisador sintático (parser) recebe a sequência de tokens gerada pelo lexer e verifica se a estrutura sintática do código segue as regras da gramática definida. Se o código estiver correto, o parser gera uma árvore sintática abstrata (AST) que representa a hierarquia das operações no código.

- **Estruturas Reconhecidas:**
  - Atribuições (`NAME = expression`)
  - Expressões matemáticas e lógicas
  - Comandos de impressão (`print`)
  - Estruturas condicionais (`if-else`)
  - Laços de repetição (`while`)

### 3. **Verificação Semântica**

O software realiza verificações semânticas básicas para garantir que o código não só seja sintaticamente correto, mas também faça sentido no contexto da linguagem. Por exemplo, verifica se todas as variáveis utilizadas foram previamente definidas.

- **Verificações Incluídas:**
  - Uso de variáveis antes de sua definição.
  - Coerência entre os tipos de dados em operações.

### 4. **Geração de Código Assembly**

A partir da árvore sintática, o simulador gera código Assembly fictício que corresponde às operações descritas no código Python original. Essa parte do processo visa ilustrar como as instruções de alto nível são traduzidas para comandos que uma máquina pode executar.

- **Instruções Geradas:**
  - Movimentação de dados (`MOV`)
  - Operações aritméticas (`ADD`, `SUB`, `MUL`, `DIV`)
  - Comparações (`CMP`)
  - Desvios condicionais (`JLE`, `JMP`, `JGE`)

### 5. **Visualização da Árvore Sintática**

O software constrói e exibe graficamente a árvore sintática gerada durante a análise sintática. Isso ajuda os usuários a visualizar a estrutura hierárquica das expressões e comandos em seu código.

- **Características da Árvore Sintática:**
  - Representação gráfica hierárquica dos nós.
  - Visualização das relações entre diferentes partes do código.

## Interface do Usuário

O software utiliza a biblioteca `Streamlit` para fornecer uma interface gráfica amigável. Os usuários podem inserir código Python em uma caixa de texto e, com o clique de um botão, iniciar o processo de análise e compilação. A interface também exibe, em uma barra lateral, os tokens identificados, a árvore sintática e o código Assembly gerado, além de quaisquer erros semânticos encontrados.

- **Componentes da Interface:**
  - Área de texto para entrada de código Python.
  - Botão para iniciar a compilação.
  - Exibição de tokens identificados.
  - Visualização da árvore sintática.
  - Exibição do código Assembly gerado.
  - Mensagens de erro sintático e semântico.

## Conclusão

Este simulador é uma ferramenta poderosa para o ensino e aprendizado de conceitos fundamentais de compiladores. Ele oferece uma abordagem prática e interativa, permitindo que os alunos experimentem diretamente com a análise léxica, análise sintática, e geração de código Assembly, proporcionando uma compreensão mais profunda desses processos.
