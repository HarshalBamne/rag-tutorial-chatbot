# rag-tutorial-v2

If running this code on colab please run the requirements.txt first and install all dependencies.

Next run the below steps in sequence
1) !npm install localtunnel
2) !curl -fsSL https://ollama.com/install.sh | sh
3) %load_ext colabxterm
4) %xterm
    This will open a terminal in colab, in which we can run the command "ollama serve"
5) Once the server is running in the above terminal we can pull the required ollama models by running "!ollama pull <model_name>" in the next cell
6) Run !ollama list to see the list of available models
7) Run !wget -q -O - ipv4.icanhazip.com to get the ipv4 address which will be needed later
8) Run !streamlit run app.py &>/content/logs.txt &
9) Run !npx localtunnel --port 8501, which will give a link, click open the link and in the password put the ip address generated in step 7