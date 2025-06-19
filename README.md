📊 Contador de Veículos com IA e Interface Gráfica
Aplicação de visão computacional desenvolvida em Python para detecção, rastreamento e contagem de veículos em vídeos. O sistema utiliza modelos YOLO (You Only Look Once) e oferece uma interface gráfica moderna e intuitiva construída com CustomTkinter.

✨ Funcionalidades Principais
Interface Gráfica Moderna: UI elegante e amigável com temas claro e escuro.

Contagem por Áreas: Defina áreas poligonais de "Entrada" e "Saída" para uma contagem precisa do fluxo de veículos.

Editor de Áreas Avançado:

Edição em Tempo Real: Arraste e solte os pontos de uma área para ajustá-la sem precisar recomeçar.

Carregamento de Áreas: Carregue e edite áreas previamente salvas, otimizando o tempo de configuração.

Interface Visual: Instruções claras e feedback visual direto na tela de desenho.

Suporte a Múltiplas Fontes: Processe vídeos de arquivos locais (.mp4, .avi) ou de streams online (URLs).

Seleção de Modelos YOLO: Escolha entre diferentes versões do YOLO (v5, v8) para balancear entre velocidade e precisão.

Download Automático de Modelos: O aplicativo baixa automaticamente o modelo YOLO selecionado caso ele não exista localmente.

Relatórios e Saída de Vídeo: Gera um arquivo de texto com o relatório da contagem e um vídeo de saída com as detecções e trilhas desenhadas.

🚀 Como Executar o Projeto
Siga os passos abaixo para colocar o aplicativo em funcionamento.

Pré-requisitos
Python 3.8+

Git (para clonar o repositório)

FFmpeg (recomendado para melhor compatibilidade com diferentes formatos de vídeo)

Instalação
Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

Crie um ambiente virtual (recomendado):

python -m venv venv

No Windows, ative com: .\venv\Scripts\activate

No macOS/Linux, ative com: source venv/bin/activate

Instale as dependências:

pip install -r requirements.txt

Execução
Com as dependências instaladas, execute o aplicativo principal:

python App.py

📖 Como Usar a Aplicação
Definir Áreas de Contagem:

Clique em "Definir Áreas de Contagem".

Selecione um vídeo para usar como fundo.

Siga as instruções na tela para desenhar as áreas de Entrada e Saída. Use o clique esquerdo para adicionar/arrastar pontos e o direito para finalizar cada área.

Pressione S para salvar. As áreas serão salvas no arquivo resultados/areas.json.

Iniciar a Contagem:

Na tela principal, escolha a fonte do vídeo (Arquivo Local ou URL).

Selecione o modelo YOLO que deseja utilizar.

Clique em "Iniciar Contagem".

Visualizar Resultados:

Após o processamento, uma mensagem informará onde o relatório (.txt) e o vídeo de saída (.mp4) foram salvos.

Use o botão "Exibir Último Relatório" para ver os resultados da contagem diretamente na aplicação.