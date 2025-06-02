import os
from dotenv import load_dotenv
from advisor.data_fetcher import ExaDataFetcher
from advisor.vector_store import VectorStore
from advisor.llm_interface import OpenAIInterface
from advisor.advisor import FantasyIPLAdvisor
from langsmith import Client, traceable

load_dotenv() 


def setup_langsmith():
    """Setup LangSmith configuration"""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    
    client = Client()
    return client

@traceable(name="fantasy_ipl_session", run_type="chain")
def run_interactive_session(advisor):
    """Run interactive session with the advisor"""
    print("Fantasy IPL Advisor - Ask me anything about IPL players, stats, or strategies!")
    print("Type 'exit' to quit, 'stats' to see vector store statistics.")
    
    session_queries = []
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
        elif query.lower() == 'stats':
            stats = advisor.vector_store.get_stats()
            print(f"\nVector Store Statistics: {stats}")
            continue
        
        try:
            response_data = advisor.get_advice(query)
            
            print(f"\nAdvisor: {response_data['response']}")
            print(f"Confidence Score: {response_data['confidence_score']:.2f}")
            print(f"Query Type: {response_data['query_type']}")
            print(f"Context Sources: {response_data['context_sources']}")
            
            session_queries.append({
                "query": query,
                "confidence_score": response_data['confidence_score'],
                "query_type": response_data['query_type']
            })
            
           
            feedback_score = input("\nRate this response (1-5, or press Enter to skip): ")
            if feedback_score and feedback_score.isdigit():
                score = int(feedback_score)
                if 1 <= score <= 5:
                   
                    print(f"Thank you for rating: {score}/5")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return {
        "total_queries": len(session_queries),
        "average_confidence": sum(q['confidence_score'] for q in session_queries) / len(session_queries) if session_queries else 0,
        "query_types": [q['query_type'] for q in session_queries]
    }

def main():
    """Main function with LangSmith integration"""
    
    langsmith_client = setup_langsmith()
    print("LangSmith tracing initialized.")
    
    exa_fetcher = ExaDataFetcher(api_key=os.getenv("EXA_API_KEY"))
    vector_store = VectorStore()
    llm = OpenAIInterface(api_key=os.getenv("OPENAI_API_KEY"))
    
    advisor = FantasyIPLAdvisor(exa_fetcher, vector_store, llm)
    
    print("Initializing advisor with latest data...")
    refresh_result = advisor.refresh_static_data()
    print(f"Data refresh complete: {refresh_result}")
    
    session_summary = run_interactive_session(advisor)
    
    print(f"\nSession Summary:")
    print(f"Total queries: {session_summary['total_queries']}")
    print(f"Average confidence: {session_summary['average_confidence']:.2f}")
    print(f"Query types: {set(session_summary['query_types'])}")
    
    print("\nThank you for using Fantasy IPL Advisor!")
    print("Check your LangSmith dashboard for detailed tracing and analytics.")

if __name__ == "__main__":
    main()