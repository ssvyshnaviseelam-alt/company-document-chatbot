from rag.pipeline import search_documents


# --------------------------------
# Evaluation test cases
# --------------------------------

test_cases = [

    {
        "question": "How many vacation days do employees receive?",
        "expected_source": "employee_handbook.pdf"
    },

    {
        "question": "How many paid sick days do employees receive?",
        "expected_source": "leave_policy.pdf"
    },

    {
        "question": "How many days per week can eligible employees work remotely?",
        "expected_source": "employee_handbook.pdf"
    },

    {
        "question": "How many company-paid holidays do employees receive?",
        "expected_source": "employee_handbook.pdf"
    },

    {
        "question": "What is the company's stock symbol?",
        "expected_source": None
    },
    {
    "question": "What are the normal business hours?",
    "expected_source": "employee_handbook.pdf"
    },

    {
        "question": "How many sick days are provided each year?",
        "expected_source": "leave_policy.pdf"
    },

    {
        "question": "How far in advance should vacation normally be requested?",
        "expected_source": "leave_policy.pdf"
    },

    {
        "question": "Can employees work remotely?",
        "expected_source": "employee_handbook.pdf"
    },

    {
        "question": "What expenses are covered during business travel?",
        "expected_source": "travel_policy.pdf"
    }
]


# --------------------------------
# Run evaluation
# --------------------------------

print("\n==============================")
print("RAG EVALUATION")
print("==============================")


retrieval_passed = 0
total_tests = len(test_cases)


for i, test_case in enumerate(test_cases):

    question = test_case["question"]
    expected_source = test_case["expected_source"]

    print("\n------------------------------")
    print(f"Test Case {i + 1}")
    print("------------------------------")

    print(f"Question: {question}")

    # --------------------------------
    # Retrieve documents
    # --------------------------------

    results = search_documents(
        question,
        top_k=3
    )

    retrieved_sources = [
        result["source"]
        for result in results
    ]

    print(f"Retrieved sources: {retrieved_sources}")

    # --------------------------------
    # Show distances
    # --------------------------------

    if results:

        best_distance = results[0]["distance"]

        print(
            f"Best distance: {best_distance:.4f}"
        )

        for result in results:

            print(
                f"  {result['source']} "
                f"| Page {result['page']} "
                f"| Distance {result['distance']:.4f}",
                f"| Score {result['score']:.4f}"
            )

    else:

        best_distance = None

        print("No documents retrieved.")

    # --------------------------------
    # Check retrieval
    # --------------------------------

    if expected_source is None:

        # For questions outside the documents,
        # we expect no relevant document.

        retrieval_pass = (
            best_distance is None
            or best_distance > 1.0
        )

    else:

        retrieval_pass = (
            expected_source in retrieved_sources
        )

    # --------------------------------
    # PASS / FAIL
    # --------------------------------

    if retrieval_pass:

        print("Retrieval: PASS")

        retrieval_passed += 1

    else:

        print("Retrieval: FAIL")


# --------------------------------
# Final evaluation
# --------------------------------

retrieval_accuracy = (
    retrieval_passed / total_tests
) * 100


print("\n==============================")
print("EVALUATION RESULT")
print("==============================")

print(
    f"Retrieval Accuracy: "
    f"{retrieval_accuracy:.2f}%"
)

print(
    f"Passed: {retrieval_passed}/{total_tests}"
)