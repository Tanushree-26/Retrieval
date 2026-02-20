import time
import numpy as np

# Ground Truth Data (Query -> Expected content substrings)
BENCHMARK_QUERIES = [
    {
        "query": "What are the primary compliance requirements for this organization?",
        "ground_truth": ["regulatory standards", "legal requirements", "adherence", "compliance framework"]
    },
    {
        "query": "What is the standard procedure for reporting a policy violation?",
        "ground_truth": ["reporting line", "compliance officer", "incident report", "within 24 hours"]
    },
    {
        "query": "What mandatory training must all personnel complete?",
        "ground_truth": ["The Service Excellence team will ensure ongoing training is provided to Operations staff for identifying claims that involve third parties.", "annual training", "onboarding requirements", "Training and Education (existing skills are insufficient to restore the worker to suitable employment."]
    },
    {
        "query": "What are the protocols for maintaining operational safety and security?",
        "ground_truth": ["safety guidelines", "security measures", "risk mitigation", "standard operating procedures"]
    },
    {
        "query": "How is sensitive or proprietary data protected under current policies?",
        "ground_truth": ["confidentiality", "data encryption", "access controls", "Need to Know"]
    },
    {
        "query": "What documentation is required for a successful internal audit?",
        "ground_truth": ["All written information obtained as a result of a disclosure or its review or investigation, must be included in the disclosure file", " When requesting an audit, the WCB will notify an employer verbally or in writing and will provide at least five business days’ notice", 
                            "verification documents", "compliance evidence"]
    },
    {
        "query": "What disciplinary measures apply to compliance non-conformity?",
        "ground_truth": ["corrective action", "written warning", "suspension", "termination of contract"]
    },
    {
        "query": "How are whistleblower reports or internal grievances handled?",
        "ground_truth": ["Where intentional avoidance has been determined, the Manager of Employer Services will also refer the employer’s.", "no-retaliation policy", "Internal Audit investigation reports. The working papers or notes of a fraud investigator of Internal Audit", "grievance committee"]
    }
]

def calculate_metrics(retrieved_chunks, ground_truth_substrings, k):
    """
    Calculates Precision@k and Recall@k.
    Relevance is determined if ANY ground_truth substring is present in the chunk.
    """
    retrieved_k = retrieved_chunks[:k]
    relevant_retrieved = 0
    
    # Check relevance for each retrieved chunk
    is_relevant_list = []
    for item in retrieved_k:
        chunk_text = item['chunk']
        is_relevant = any(gt.lower() in chunk_text.lower() for gt in ground_truth_substrings)
        is_relevant_list.append(is_relevant)
        if is_relevant:
            relevant_retrieved += 1
            
    # Precision@k = (Relevant Retrieved) / k
    precision = relevant_retrieved / k if k > 0 else 0
    
    # Recall@k = (Relevant Retrieved) / (Total Relevant in Corpus)
    # Since we don't know the exact total relevant in corpus without full annotation, 
    # and we usually have 1 or 2 distinct facts, we'll approximate Total Relevant 
    # by the number of ground truth substrings we're looking for (assuming each maps to a chunk)
    # or just assume there's at least 1 relevant chunk. 
    # For this simplified benchmark, let's assume if we found ANY relevant info, we found the "answer".
    # But strictly, Recall = (TP) / (TP + FN). 
    # Let's assume there is exactly 1 ideal chunk for each query for simplicity in this demo, 
    # unless we find multiple.
    total_relevant = 1 
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
    recall = min(recall, 1.0) # Cap at 1.0
    
    return precision, recall

def run_benchmark(retriever, k=3):
    """
    Runs the benchmark on both retrieval techniques.
    """
    results = []
    
    for item in BENCHMARK_QUERIES:
        query = item['query']
        ground_truth = item['ground_truth']
        
        # 1. Vector Retrieval
        start_time = time.time()
        vector_results = retriever.vector_search(query, k)
        vector_latency = time.time() - start_time
        
        v_prec, v_rec = calculate_metrics(vector_results, ground_truth, k)
        
        # 2. Hybrid Retrieval
        start_time = time.time()
        hybrid_results = retriever.hybrid_search(query, k)
        hybrid_latency = time.time() - start_time
        
        h_prec, h_rec = calculate_metrics(hybrid_results, ground_truth, k)
        
        results.append({
            "query": query,
            "vector": {
                "precision": v_prec,
                "recall": v_rec,
                "latency": vector_latency
            },
            "hybrid": {
                "precision": h_prec,
                "recall": h_rec,
                "latency": hybrid_latency
            }
        })
        
    return results
