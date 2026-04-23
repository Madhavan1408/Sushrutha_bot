from loader import load_folder, load_urls
from langchain_text_splitters import RecursiveCharacterTextSplitter
URLS = [
        "https://medlineplus.gov/aorticaneurysm.html",
        "https://www.nhlbi.nih.gov/health/aortic-aneurysm/symptoms",
        "https://www.nhlbi.nih.gov/health/ards",
        "https://www.nhlbi.nih.gov/health/alpha-1-antitrypsin-deficiency",
        "https://www.nhlbi.nih.gov/health/anemia",
        "https://www.nhlbi.nih.gov/health/angina",
        "https://www.nhlbi.nih.gov/health/antiphospholipid-syndrome",
        "https://www.nhlbi.nih.gov/health/aortic-aneurysm",
        "https://www.nhlbi.nih.gov/health/anemia/aplastic-anemia",
        "https://www.nhlbi.nih.gov/health/arrhythmias",
        "https://www.nhlbi.nih.gov/health/asthma",
        "https://www.nhlbi.nih.gov/health/atherosclerosis",
        "https://www.nhlbi.nih.gov/health/atrial-fibrillation",
        "https://www.nhlbi.nih.gov/health/bleeding-disorders",
        "https://www.nhlbi.nih.gov/health/blood-cholesterol",
        "https://www.nhlbi.nih.gov/health/clotting-disorders",
        "https://www.nhlbi.nih.gov/health/blood-tests",
        "https://www.nhlbi.nih.gov/health/bronchitis",
        "https://www.nhlbi.nih.gov/health/cardiac-arrest",
        "https://www.nhlbi.nih.gov/health/cardiac-catheterization",
        "https://www.nhlbi.nih.gov/health/cardiogenic-shock",
    ]

pdf_docs=load_folder(r"D:\AI Course\medial_bot\data")
url_docs=[]
for url in URLS:
    print(f"Loading URL : {url}")
    docs=load_urls([url])
    url_docs.extend(docs)
medical_data=pdf_docs + url_docs
print(f"Loaded {len(medical_data)} documents from PDFs and URLs combined.")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs=splitter.split_documents(medical_data)
print(f"Split into {len(split_docs)} chunks of text.")