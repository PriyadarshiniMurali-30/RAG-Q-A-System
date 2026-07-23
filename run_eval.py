import os
from dotenv import load_dotenv
from groq import Groq
from query import answer_question  # reusing your existing pipeline!
from eval_questions import EVAL_QUESTIONS

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def judge_answer(question, expected_answer, actual_answer):
    """Use an LLM to judge if the actual answer matches the expected answer in meaning."""
    judge_prompt = f"""You are grading an AI system's answer against an expected answer.

Question: {question}
Expected answer: {expected_answer}
AI's actual answer: {actual_answer}

Does the AI's answer correctly convey the expected answer (even if worded differently)?
Reply with ONLY one word: "CORRECT" or "INCORRECT"."""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "CORRECT" in verdict


def run_evaluation():
    results = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        expected = item["expected_answer"]

        print(f"\nAsking: {question}")
        actual_answer = answer_question(question)["answer"]

        is_correct = judge_answer(question, expected, actual_answer)
        results.append({
            "question": question,
            "expected": expected,
            "actual": actual_answer,
            "correct": is_correct
        })

        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"{status}")

    # --- Summary report ---
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = (correct_count / total) * 100

    print("\n" + "=" * 50)
    print(f"EVAL RESULTS: {correct_count}/{total} correct ({accuracy:.1f}%)")
    print("=" * 50)

    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(f"{status} {r['question']}")
        if not r["correct"]:
            print(f"   Expected: {r['expected']}")
            print(f"   Got: {r['actual'][:150]}...")

    return results


if __name__ == "__main__":
    run_evaluation()