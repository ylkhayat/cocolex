dataset_to_system_prompt = {
    "clerc": "You are a helpful legal professional.",
    "echr_qa": "You are an ECHR legal expert tasked to answer a question.",
    "cuad": (
        "You are a helpful legal professional analyzing clauses in a contract. "
        "Answer the following question based on the following contract. Please output No relevant information in the contract, if the answer to question is not present in contract."
    ),
    "obli_qa": (
        "You are a regulatory compliance and legal expert reviewing regulatory documents. "
        "Do not directly answer the user's question or provide any interpretation. "
        "Instead, only provide exact excerpts from the regulatory text that are relevant to the question, "
        "verbatim and without additional commentary."
    ),
    "oal_qa": (
        "You are an Australian legal expert providing answers to legal questions based on referenced cases and judgments. "
        "Avoid making assumptions or interpretations beyond the provided information."
    )
}

dataset_to_context_prefix = {
    "clerc": (
        "Below are reference cases provided for factual accuracy. When generating content, you must "
        "reference and cross-check the relevant details with the provided reference texts by their "
        "reference IDs. (e.g., {joined_retrieved_ids}). Your output must align with these references."
    ),
    "echr_qa": (
        "The following documents were retrieved and should help you answer the question. "
        "You must refer to these documents when answering the question. (e.g., {joined_retrieved_ids}). "
        "Valid citation formats: [{single_retrieved_id}] or [{joined_retrieved_ids}]. "
    ),
    "cuad": (
        "Below are the only contract clauses that serve as your knowledge base."
    ),
    "obli_qa": (
        "Below are the only regulatory documents that serve as your knowledge base."
    ),
    "oal_qa": (
        "The following Australian legal cases and documents were retrieved to assist in answering the question. "
        "You must reference these documents in your answer using their ids (e.g., {joined_retrieved_ids})."
    )
}

dataset_to_prompt_prefix = {
    "clerc": (
        "Continue to write the following case in the style of my writeup. Your answer should range "
        "from 100 to 400 words. Make your answer concise, and avoid redundant language and assumptions. "
        "Below is what I have written so far:"
    ),
    "echr_qa": (
        "Answer the following question using the retrieved documents. "
        "Reuse the language from the documents! "
        "Cite relevant documents at the end of a sentence! "
        "Accepted formats: sentence [citation(s)]. "
        "You must follow the [Doc i] format! Do NOT use the case names or paragraph numbers to cite documents! "
        "You should NOT provide a list of all used citations at the end of your response!\n\n"
        "Question: "
    ),
    "cuad": (
        "Answer the following prompt based on the above contract."
    ),
    "obli_qa": (
        "Question: "
        "Extract only the directly relevant regulatory text segments related to the user's query. "
        "Do not answer or provide explanations, only produce the exact text segments that apply."
    ),
    "oal_qa": (
        "Answer the following question using the referenced Australian legal cases and documents. "
        "Question: "
    )
}

dataset_to_prompt_suffix = {
    "clerc": "",
    "echr_qa": "\nAnswer: ",
    "cuad": (
        "\n"
        "If no relevant text explicitly addressing the category is found, respond only with 'No relevant information found in the document.' "
        "Otherwise, begin your response with 'Highlights:\\n' followed by the exact text segments as they appear in the contract."
        "\n"
        "Answer: "
    ),
    "obli_qa": (
        "\n"
        "Answer the question through providing the exact relevant text segments as they appear in the regulatory document. "
        "\n"
        "Answer: "
    ),
    "oal_qa": (
        "\n"
        "Answer the question using the provided Australian legal cases and documents. "
        "\n"
        "Answer: "
    )
}