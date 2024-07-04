from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.bot.data_models.grade_answer import GradeAnswer
from src.bot.data_models.grade_documents import GradeDocuments
from src.bot.data_models.grade_hallucinations import GradeHallucinations
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.bot.data_models.grade_question import GradeQuestionType
from src.bot.graph.graph_state import GraphState
from langgraph.graph import END, StateGraph, START
import streamlit as st
def start_ug_buddy():

    system_prompt = """You are an assistant that determines if a user query is a pleasantry or requires a factual response. 
    If the query is a pleasantry, grade it as 'pleasantry'. If it requires a factual response, grade it as 'factual'.
    Examples of pleasantries include 'Hello', 'How are you?', 'Good morning', etc.
    Examples of factual questions include 'What are the cut-off points for Computer Science?', 'What scholarships are available?', etc."""

    question_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: \n\n {question}"),
        ]
    )

    question_grader = question_grader_prompt | llm.with_structured_output(GradeQuestionType)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    preprompter = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = preprompter | llm | StrOutputParser()


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            if not documents:
                return "no docs"
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    def grade_question_type(state):
        """
        Determines whether the question is a pleasantry or requires a factual response.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---GRADE QUESTION TYPE---")
        question = state["question"]
        score = question_grader.invoke({"question": question})
        grade = score.binary_score

        if grade == "pleasantry":
            print("---DECISION: PLEASANTRY---")
            return "pleasantry_response"
        else:
            print("---DECISION: FACTUAL---")
            return "retrieve"

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        print(f"question: {question}")
        print(str(state))
        return {"documents": documents, "question": question}
    
    def pleasantry_response(state):
        """
        Generates a response for pleasantries.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates the state with a pleasantry response
        """
        print("---RESPOND TO PLEASANTRY---")
        question = state["question"]
        response = llm.invoke({"question": question})
        return {"generation": response, "question": question}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}
    
    def transform_query(state):
        """
        Transforms the query for a better match.

        Args:
            state (GraphState): The current graph state

        Returns:
            GraphState: Updates the state with transformed query and increment tries
        """
        print("---TRANSFORM QUERY---")
        state["transformed_tries"] += 1

        if state["transformed_tries"] > 5:
            response = llm.invoke({"question": "Sorry, it seems your question has no references in our data source. Can I help you with something else?"})
            state["generation"] = response
            return END
        else:
            transformed_query = llm.invoke({"question": state["question"]})
            state["question"] = transformed_query
            return state
    workflow = StateGraph(GraphState)
    workflow.add_node("grade_question_type", grade_question_type)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("pleasantry_response", pleasantry_response)

    workflow.add_edge(START, "grade_question_type")
    workflow.add_edge("grade_question_type", "retrieve")
    workflow.add_conditional_edges(
        "grade_question_type",
        grade_question_type,
        {
            "pleasantry_response": "pleasantry_response",
            "factual": "retrieve",
        }
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
            "no docs": END
        },
    )
    app =  workflow.compile()
    return app
print('starting vectorstore')
vectorstore = PineconeVectorStore(
    embedding=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]),
    index_name=st.secrets["PINECONE_INDEX_NAME"],
)
retriever = vectorstore.as_retriever()
print('got retriever')
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
pre_tune_prompt = """You are UG Buddy, an AI chatbot designed to help prospective students navigate the University of Ghana (UG). 
            Your goal is to provide user-centric and reliable information, acting as a knowledgeable and approachable peer. 
            You aim to make the exploration of UG's offerings more accessible, personalized, and engaging. 
            You assist with academic programs, admissions, financial aid, and student life information. Here's a brief summary of your knowledge:
            1. Academic Programs:
            - Information about various undergraduate programs.
            - Cut-off points for different programs.
            - Course details, duration, and structure.
            2. Admissions:
            - Admission requirements and procedures.
            - Application deadlines and important dates.
            - Contact information for the admissions office.
            3. Financial Aid:
            - Scholarships and financial aid opportunities.
            - Eligibility criteria and application process.
            - Resources for budgeting and managing finances.
            4. Student Life:
            - Campus facilities and resources (libraries, sports, etc.).
            - Housing options and accommodations.
            - Extracurricular activities and student organizations.
            5. General Information:
            - Location and transportation options.
            - Campus safety and support services.
            - Important contact information and FAQs.
            When interacting with users, always aim to provide clear, concise, and accurate information. Personalize your responses based on the user's questions and needs. Here are some example interactions:
            ---
            User: What are the cut-off points for the Bachelor of Science in Computer Science?
            UG Buddy: The cut-off point for the Bachelor of Science in Computer Science at the University of Ghana is typically around [insert current cut-off point]. However, this can vary each year based on the number of applicants and available slots. It's best to check the latest information on the UG admissions website or contact the admissions office directly at [admissions contact information].
            ---
            User: Can you tell me about the financial aid options available for international students?
            UG Buddy: Sure! The University of Ghana offers various financial aid options for international students, including scholarships and grants. Some of the notable scholarships include [list of scholarships]. To be eligible, you typically need to meet certain academic and financial criteria. The application process usually involves [brief overview of the process]. For more detailed information, you can visit the financial aid office website or contact them at [financial aid contact information].
            ---
            User: What kind of extracurricular activities can I participate in at UG?
            UG Buddy: The University of Ghana offers a wide range of extracurricular activities, including student clubs, sports teams, and cultural organizations. Whether you're interested in joining a debate club, participating in sports like soccer or basketball, or exploring your artistic talents in a drama or music group, there's something for everyone. These activities are a great way to meet new people, develop new skills, and enhance your university experience. You can find a list of student organizations and their contact details on the UG student affairs website.
            ---
            Feel free to ask me any questions about the University of Ghana. I'm here to help you make informed decisions and feel more connected to the university.
            """
llm.invoke(pre_tune_prompt)
app = start_ug_buddy()