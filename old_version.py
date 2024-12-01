from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Literal, Dict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable


CATEGORIES = Literal[
    "Visual Art",
    "Literature",
    "History",
    "Music",
    "Architecture",
    "Artificial Intelligence in General",
    "Generative Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Bayesian Statistics",
    "Causal Inference",
    "Biology in General",
    "Genetics",
    "Bioinformatics",
    "Mathematical Biology",
    "Blockchain",
    "Computer & Programming in General",
    "Cosmology, Quantum Mechanics",
    "Data Science & Statistics in General",
    "Medical Statistics",
    "Structural Equation Modeling",
    "Time Series Forecasting",
    "Big Data Analysis",
    "Engineering in General",
    "Futurology",
    "Science, General",
    "Physics in General",
    "Chemistry in General",
    "Internet of Things",
    "Language, Linguistics",
    "Mathematics in General",
    "Calculus",
    "Linear Algebra",
    "Geometry",
    "Topology",
    "Number Theory",
    "Mathematical Analysis",
    "Medicine in General",
    "Pain",
    "Nanotechnology",
    "Network Science",
    "Complex Systems",
    "Neuroscience",
    "Philosophy",
    "Psychiatry in General",
    "Psychopharmacology",
    "Psychotherapy, psychoanalysis",
    "Psychology",
    "Spirituality, Religion",
    "Quantum Computing",
    "Others",
]


class Category(BaseModel):
    """Category that best represents the given book's subject"""

    subject: CATEGORIES = Field(
        ..., description="The most representative subject of the book"
    )


SYSTEM_MESSAGE = """The file name entered by the "user" includes 
the title and subtitle of the academic book.
Select only one of the following optional categories that best expresses the theme of this book. 
The "in General" option can only be selected when there is no other more specific category that applies.
If none of the provided options applies, just select "Others"

IMPORTANT: NEVER use other categores not included in the below Optional Categories.

Optional Categories
=====================
    "Visual Art",
    "Literature",
    "History",
    "Music",
    "Architecture",
    "Artificial Intelligence in General",
    "Generative Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Bayesian Statistics",
    "Causal Inference",
    "Biology in General",
    "Genetics",
    "Bioinformatics",
    "Mathematical Biology",
    "Blockchain",
    "Computer & Programming in General",
    "Cosmology, Quantum Mechanics",
    "Data Science & Statistics in General",
    "Medical Statistics",
    "Structural Equation Modeling",
    "Time Series Forecasting",
    "Big Data Analysis",
    "Engineering in General",
    "Futurology",
    "Science, General",
    "Physics in General",
    "Chemistry in General",
    "Internet of Things",
    "Language, Linguistics",
    "Mathematics in General",
    "Calculus",
    "Linear Algebra",
    "Geometry",
    "Topology",
    "Number Theory",
    "Mathematical Analysis",
    "Medicine in General",
    "Pain",
    "Nanotechnology",
    "Network Science",
    "Complex Systems",
    "Neuroscience",
    "Philosophy",
    "Psychiatry in General",
    "Psychopharmacology",
    "Psychotherapy, psychoanalysis",
    "Psychology",
    "Spirituality, Religion",
    "Quantum Computing",
    "Others",
"""


def categorize(path: Path, chain: Runnable) -> Dict:
    """Categorizes the book based on its title and subtitle"""
    # Load the file name from the given path
    files = [f for f in path.iterdir() if f.is_file()]

    output = {}
    with open("./error.log", "a", encoding="utf8") as f:
        for file_path in tqdm(files):
            try:
                response = chain.invoke({"file_name": file_path.name})
                output[file_path.name] = response
            except Exception as e:
                f.write(f"Failed to categorize {file_path.name}\n")
                f.write(f"reason: {e}")
    with open("./categorized_result.json", "w", encoding="utf8") as f:
        to_dump = {k: v.subject for k, v in output.items()}
        print(to_dump)
        json.dump(to_dump, f, indent=4)
    return output


def move_files(output: Dict, path: Path):
    with open("./error.log", "a", encoding="utf8") as f:
        for file_name, category in output.items():
            try:
                subject = category.subject
                subject_dir = path / subject
                subject_dir.mkdir(parents=True, exist_ok=True)
                (path / file_name).rename(subject_dir / file_name)
            except Exception as e:
                f.write(f"Failed to move {file_name}\n")
                f.write(f"reason: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify books into appropriate categories and move the files in the created directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--directory", default=".", help="directory of book collection"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o",
        help="GPT model to use (default=gpt-4o)",
    )
    args = parser.parse_args()
    path = Path(args.directory)
    model_name = args.model

    llm = ChatOpenAI(model="gpt-4o").with_structured_output(Category)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_MESSAGE), ("user", "{file_name}")]
    )
    chain = prompt | llm

    output = categorize(path=path, chain=chain)
    print("Categorization Completed.")
    move_files(output=output, path=path)
