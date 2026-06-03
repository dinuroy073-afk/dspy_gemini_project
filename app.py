import os
import dspy
from dotenv import load_dotenv

# Load environment variables from the project .env file
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

# Read Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add it to your .env file."
    )

# Configure DSPy with a Gemini model that supports the current v1beta endpoint
lm = dspy.LM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key
)

dspy.configure(lm=lm)


# DSPy Signature
class TechQA(dspy.Signature):
    """
    Answer technical interview questions clearly and accurately.
    """

    question = dspy.InputField(
        desc="Technical question asked by the user"
    )

    answer = dspy.OutputField(
        desc="Simple and detailed technical answer"
    )


# DSPy Module
class InterviewAssistant(dspy.Module):

    def __init__(self):
        super().__init__()

        # Chain of Thought Reasoning
        self.generate_answer = dspy.ChainOfThought(TechQA)

    def forward(self, question):

        result = self.generate_answer(question=question)

        return dspy.Prediction(
            answer=result.answer
        )


# Main Application
def main():

    assistant = InterviewAssistant()

    print("\n===================================")
    print("   DSPy + Gemini AI Assistant")
    print("===================================\n")

    print("Type 'exit' to stop the application.\n")

    while True:

        question = input("Ask Technical Question: ")

        if question.lower() == "exit":
            print("\nGoodbye!\n")
            break

        try:
            response = assistant(question=question)

            print("\n========== ANSWER ==========\n")
            print(response.answer)

        except Exception as e:
            print("\nError:", e)

        print("\n============================\n")


if __name__ == "__main__":
    main()
