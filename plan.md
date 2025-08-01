# Plan

# Context
We have a dataset of customer conversations for a new customer called "Brown Box". We want to create an ontology/types of customer intents for these conversations based on ~1k example convos available Will be useful for:
- Deploying different agents for different customer intents
- Avoid certain types of convos for agents, like "fraud" (maybe to pass on to human agents)

# Task: 
We need to create... 
1. An ontology to categorise conversations into different types of customer intents (to create different agents for them)
2. A mechanism to evaluate if the ontology covers all customer intents and doesn't need to be improved further.

## Challenge 1: Customer Intent Ontology

The goal is to create a list of customer intents that can be used to create specific agents that handle each or a few related customer intents well.

That means, for the ontology to be good:
- Mutually exclusive (ME): Are the customer-intents non-overlapping? Important to ensure
- Collectively exhaustive (CE): Does it cover every customer intent noticed in customer conversations.
- Intent-oriented labels: Labels should be based on what the customer's need is (Not reason for contact/product lines/other characteristics of the conversation)

### Brainstorming: Challenge 1
- What about multi-intent conversation? Should still be fine cuz agents can handle multiple intents and switch to different agents.
- We have to assume here there could be multiple intents for the same conversation and when labelling we should be careful to not force it into just one intention.
- I should read a bunch of example conversations to get some better ideas too
- I am guessing LLMs are the best way to generate these customer intents - can do an iterative loop to generate an ontology of customer intents.
- Review customer intents, definitions and examples to see if the ontology seems right. Manually refine it too for accurcy.
- Then use it to label the dataset with the ontology. Caveats: Allow convos to be tagged with multiple-intents and have an 'other' category, potentially with reasons on why this was put in an other category.
- Refine it again for fitting 'other' categories in the ontology.
---
Lets stick to this approach, with one modification. There are three tasks: 
1. Generate customer-intent categories (Clustering + Eval Clust.)
2. Classify conversations based on categories (Classifying + Eval Class.)
3. Evaluate if classification and clustering are correct (Iteration)

For (1), earlier, we were using LLMs to iteratively generate the classification categories for the ontology. Now two modifications:
- We'll generate ontology categories + defenitions based on the first 300 conversations.
- We'll manually adjust with random cross-checking + embedding clusters
- Label the 300 with the ontology + 'other' category. Adjust it to remove other categories manually.
- Evaluate classification correctness, ME (manual check), CE ('other' category < 5%)
- After fitting all 'other' categories in the ontology, classify 1k conversations + evaluate classification and ontology. 

#### Alternative Approaches

- Could do a manual approach, but its 1k files long so not logical to do it manually and at scale.
- I could use an ML-based clustering mechanism but I'll have to train it from scratch to categorise for customer-intents. Following that I need to build another ML-classifier model to classify each conversation based on examples. Either way need to spend time on a lot of manual labelling, training and then using it. Not worth it for an ontology that frequently changes.
- LLMs definitely are the best choice because they can easily be modified to look for different characteristics in a conversation and they can be used both for ontology generation and classification. No training involved, just prompt changes, so works perfectly with frequent changes with ontology and across different customers.

**Important**: We should think about why ME is a necessary metric to judge the ontology. I get why CE is, cuz the agents need to have a plan on how to handle each customer intent. and I also get why action-oriented customer-intent labels are, cuz its better to categorise based on what the user wants to achieve rather than keywords or product line because the primary purpose of customer service agents are to fulfill customer's needs, so its best to build agents that perform well in fulfilling customer intents. Plus, its been given that customer intent is what they want to categorise conversations based on, so no need to question the premise when you agree with it. But with the 1st metric of ME, I mean if customer-intents are not mutually exclusive, then there is no reason to categorise conversations in the first place. And the reason for categorising conversations is its easier to create specialised agents that can deliver on customer needs when they are independent and are not linked to other customer intent categories.

### Brainstorming: Challenge 2

- For the 2nd Challenge of evaluating the ontology - there are two things to evaluate: Are the customer-intents non-overlapping/ME, Can the ontology reasonably categorise every conversation from the test dataset without having any new/depcrated customer-intents, Are the lables action-oriented or do they revolve around another dimension?
- For the 1st metric of ME customer-intents, we can preferably manually evaluate the ontology after every time it is changed (because when done right it wouldn't be on a daily basis). But if needed to have a health check for ME customer-intents, we can do it using an LLM calls of evaluating the ontology with a reasoning LLM.
- For the 2nd metric of CE customer-intents, we use a program to label a test-dataset using the customer-intent ontology, along with an additional category called 'other' if the program wasn't able to classify it. So if there is more than 5 or x % of 'other' conversations, then the ontology needs to be updated. We'll need to think on what is a good benchnark for ratio of 'other' conversations.


## Next Steps

- [Completed] Lets first, download the dataset, remove other columns and only have the conversations column.
- Then lets review the a few conversations in excel to get a feel for what they look like.
- Based on that lets create a program called 'create_ontology' to call an LLM to iterate over every conversation and update the ontology. Lets also get the function to label the datasets with the ontology (will be helpful for cross examining ontology with examples and refining it before proper labelling - can create two functions to either just create ontology or with labelling). We'll save the ontology to a local md file.
- After we get an initial ontology, lets review it, along with cross examinining with examples and adjust ontology until we're happy with defenitions to improve accuracy.
- Then lets create this 'label_conversations' program to label convos based on the ontology and also let it use the 'other' category and adjust ontology iteratively until 'other' category is 0%.
- Then write a programm called 'evaluate_ontology' to health check for mutual exclusivity, evaluate for collective exhaustion, and maybe intent-oreiented labelling (but 3rd can be decided later after creating first ontology set)

----

# Progress

## Step 1: Create Ontology

### To do
- Need to figure out a reliable method to gerneate ontology using LLMs first - zero shot prompting with 300 conversations, JSON schema based customer-intent category generation [SPIRES, zero shot KG]
- After we implement that and get some results, we will try using embedding clusters to improve it and also merge overlapping categories
- Then fit 'other' categories and evals on the 300 convos
- Finally we'll be done with creating Ontology
- Then for the finishing touch, we refine the create_ontology program

### Generating Ontology with LLMs

What I mean by customer-intent labels: A “customer intent taxonomy/labels” is a structured set of categories (labels) representing the different reasons customers contact support, organized—where possible—into a hierarchy (broad-to-specific), with each category having a clear, human-readable definition. Could be flat or hierarchical.

#### Reason to go for Non-CQs based approach:

When CQs don’t help much
	•	If your goal is just a list of meaningful, distinct categories, directly grounded in your data.
	•	When your dataset is large and diverse and you don’t have a clear idea what the CQs should be until after you’ve seen the main clusters of intent.
	•	If you have no subject matter experts (SMEs) to vet or prioritize which CQs really matter for business needs.

### Problems with the challenge's solution + Ideas to solve it

Problem: Firstly, Hdbscan is not yielding good results, so wanted to check if there is something I am missing.
Idea: We'll try different settings, but thats not the most important, we'll come back and try it if we want. Current better algorithm is aglommorative clustering.

More important problem: The mutual exclusivity metric is not deterministic and is too wishy washy for judging.
Idea: Let's try cosine similarity to test if it can accurately judge mutual exclusivity among categories.

For challenge #1 of creating an ontology to classify conversations based on the customer intent, these are potential problems we have with the solution:

1. 1 or 2 pairs seem to have slight overlaps, if so we need a systematic way to fix it. But we need a mechanism to properly judge mutual exclusivity first - we're gonna use human evals/llm evals for a binary test.

2. [Solved] We need a solid way to judge if the labels are mutually exclusive enough - we're gonna try cosine similarity to solve that - perfectly completed. We've identified the most mutually exclusive set of labels, which happens to have the highest coverage rate too! So now we have a deterministic way to idetify the most mutually exclusive set of labels and also judge collective exhaustivity.

For challenge #2,

1. As we mentioned above, mutual exclusivity is not being judged properly so we need a way to judge mutual exclusivity of ontologies first. And then maybe if necessary come up with a way to fix similar pairs.

2. Need a logic to judge when to stop iterating on the ontology. Simple really, lets go from first principles. Our goal is to have good resolution rates with our AI agents (CE). The other thing is

## Factors used to judge if an ontology is good enough
An ontology is good enough if it helps us achieve what we are trying to achieve with it. Our goal is to use these customer intent labels to generate specialised agents that can handle these customer intents or avoid them. So they need to:
- Be able to classify 99 out of every 100 calls accurately (coverage)
- Should be mutually exclusive enough to the extent that there are 0 duplicates, similarities are fine to describe more nuanced customer intents (duplicate check - LLM flags, cosine similarity)
- Doesn't have redundant customer intents (distribution check)

## Main thing left to do to get answer for Challege #1 and #2
- Get all 4 metrics stored for different distance thresholds and choose one.

## Problems/Solutions

1. The whole ontology creation process is a single file and the different clustering results are super unorganised. So hard to understand, harder to reuse.

**Solution:**
1. Actually before everything, lets split the file, its a lot of code. There are four things we do here - Generate customer intents list, cluster them, create ontology, then evaluate them.
2. First lets figure naming conventions and how to save init customer intents, clusters and final customer intents in an organised way.
