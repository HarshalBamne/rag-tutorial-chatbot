# rag-demo


To run in VScode
1) Store you google_api_key in the .env file
2) Install the requirements .txt file and then run streamlit run app.py --server.enableXsrfProtection false to launch the chatbot.

If running this code on colab please clone the repo, store the google_api_key, run the requirements.txt first and install all dependencies.

Next run the below steps in sequence:

1) !npm install localtunnel
2) Run !wget -q -O - ipv4.icanhazip.com to get the ipv4 address which will be needed later
3) Run !streamlit run app.py &>/content/logs.txt &
4) Run !npx localtunnel --port 8501, which will give a link, click open the link and in the password put the ip address generated in step 2