# Coroner Project

## How to Get Latest Version of Repo & Update Conda Environment

First you'll need to remove the current environment with this command:

`conda env remove --name coroner_env`

Next you'll need to download the current version of the repo, i.e., `Download ZIP` from the **code** button.

Now you need to rebuild the environment. Enter the top level directory of the **coroner-main** repo you just downloaded. Now enter the following command:

`conda env create -f environment.yml`

That's it. Now you'll be able to run the webapp like so:

`streamlit run app.py`

## Setup

Setting up all the dependencies and the environment is somewhat complex and will take quite a while, so make sure you have the time to sit down and go through everything carefully. This is unavoidable because of the nature of our project, i.e., the main requirement is that everything runs locally.

### Ollama

First you'll need to install the ollama model server so that we can locally interact with the various LLMs we'll be using. You can download and install ollama from [here](https://ollama.com/).

We'll be using at least three different LLM models in addition to the embedding model for the vector database so you'll need to pull these models from the ollama catalog to your local machine. Once you have ollama installed and running you can start by installing the embedding model.

**NB** Once installed you can type `ollama` at the terminal to see some help info.

### Embedding Model

The embedding model we are using is [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large). To install it you can use the following command:

`ollama pull mxbai-embed-large`

Once it's installed use the `ollama list` command. You should see some details about the model in the terminal.

### LLM Llama 3.2

All the LLM models we'll be using can be found [here](https://github.com/ollama/ollama/blob/main/README.md#model-library). Take note of the size of the models and the RAM requirements. The first one you'll need to install is [llama3.2](https://ollama.com/library/llama3.2) (2.0 GB) which can be installed with the following command:

`ollama pull llama3.2`

### LLM gemma3

Next install [gemma3](https://ollama.com/library/gemma3) (3.3 GB) with this command:

`ollama pull gemma3`

### LLM phi4-mini

Now install [phi4-mini](https://ollama.com/library/phi4-mini) (2.5 GB) with this command:

`ollama pull phi4-mini`

### Conda Environment

If you have the ollama model server correctly setup and have installed all the models you can next create the Python environment with conda. If you don't already have conda installed you can follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Once conda is up and running you'll need to create the environment for our project which has all the dependencies whithin it. To do so, open up a terminal and type the following command:

`conda env create -f environment.yml`

**NB** You'll need to be in the main directory of this repo for the command to work because that is where the `environment.yml` file is located. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more details. If you've done everything correctly you should now be able to enter the environment by typing `conda activate coroner_env`.

## Start Chatting

If you have successfully set up the environment you should now be able to start the application. Again, make sure you are in the main directory of this repo and you have the `coroner_env` conda environment running. Now, simply type `python chat.py` and you should be able to start chatting with one of the documents (it'll take a little while to boot the first time). 

**NB** The documents have a `.jsonl` extension because I have already pre-vectorized the chunks of each document. This makes the process a little smoother and much faster.

Here is an example interaction:

```
ajh@9000 coroner % conda activate coroner_env
(coroner_env) ajh@9000 coroner % python chat.py

Available LLMs:

  1. phi4-mini
  2. gemma3
  3. llama3.2

Enter the number of the LLM you want to use: 2

Available files:

  1. TAULELEI-Jacob-Finding.jsonl
  2. Rodier-Finding.jsonl
  3. Blood-results-redacted.jsonl
  4. Forkin-finding-2014.jsonl
  5. Baby-H-finding.jsonl
  6. Nicholls-Diver-finding.jsonl

Enter the number of the file you want to use: 2

Initializing, please wait...

Loading jsondata/Rodier-Finding.jsonl

LLM & vector store ready.

Starting chat.



--------------------------------------------------------------------------------


Ask your question (type q to quit): What is under investigation?


Question:
What is under investigation?

Answer:
The investigation is into the death of Frank Edward Rodier, which began with a suspicion based on police information and culminated in an inquest. Specifically, the investigation concluded that Frank Rodier died on 25 1975 in the waters of the Indian Ocean off Quobba Station near Red Bluff, Carnarvon, likely due to drowning.

Source 1:
  * text: "INTRODUCTION\n- 2 In my capacity as the Acting State Coroner, I determined on the basis of information provided by the WA Police in August 2023 that   there was   reasonable cause to suspect that Frank had died and that his death was a reportable death under the Act. I therefore made a direction to the Commissioner of Police; pursuant to s 23(1) of ..."
  * page: 3
  * document: data/Rodier-Finding.pdf

Source 2:
  * text: "INTRODUCTION\n- 3 On 11 October  2023 a report  prepared by Detective Sergeant Ellie Wold from the Homicide Squad Missing Person Team. In the report, Frank was confirmed to be a long term missing person, with his disappearance first reported to police at about 10.25 am on 25 1975. In 2006, a review by police had determined Frank's disappearance fell..."
  * page: 3
  * document: data/Rodier-Finding.pdf

Source 3:
  * text: "RECORD OF INVESTIGATION INTO DEATH\nI, Sarah Helen Linton; Deputy State Coroner , having investigated the death of Frank Edward RODIER with an inquest  held at Perth Coroners Court; Central Law Courts, Court 85, 501 Street; Perth, on 14 August 2024, find that the identity of the deceased person was Frank Edward RODIER and that death occurred on 25 1..."
  * page: 2
  * document: data/Rodier-Finding.pdf


--------------------------------------------------------------------------------


Ask your question (type q to quit): q


(coroner_env) ajh@9000 coroner %
```


## Evaluations

With everything up and running we'll then need to evaluate each of the models on each of the documents so that we'll have an evaluation "report" for each (document, model) pair. For an example of how to do this see `Rodier-Finding-llama3.2.md` in the `evaluations` directory. There you can see that the evaluation scheme is as follows:

- Pose a **question** that we know the correct answer to by examining the oringal document.

- Ask the **question** (i.e., give the question to the chat application).

- Get the **answer** (i.e., the output from the chat application).

- Evaluate the **correctness** and **relevance** by giving a score for the **answer** and the **sources** respectively (i.e., LOW, MED, HIGH).

We can then collate the results and report them. An example plot might look something like this:

![Example Plot](/evaluations/evaluation-example.png)

**NB** Whilst there exist purely statistical evaluation metrics for text based outputs it is not ideal to have a strictly numerical *scoring* system for this evaluation scheme. Given the small number of documents we have under consideration our approach will be a *human level evaluation*. That is, the assessment is based on: 

- **Correctness**: Are the claims generated by the model factually accurate?
- **Relevance**: Is the information relevant to the given prompt? 

Furthermore, because of the nature of our documents, we take the approach that: ["Ultimately, human experts sometimes need to verify facts. This is common in domains like medical or legal, where every statement might need verification."](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-metrics-A-comprehensive-guide-for-large-language-models--VmlldzoxMjU5ODA4NA#factuality-and-faithfulness-–-is-the-content-correct-and-grounded-in-truth?) 

Finally, the *scores* given are a product of your *human* understanding which is established through careful examination of the original documents. For more details on these considerations refer to [this](https://rewirenow.com/en/resources/blog/how-to-evaluate-the-quality-of-your-large-language-model-output-before-deploying/).



