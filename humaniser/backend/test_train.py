import fitz
import requests
import os
import json
import time

# Sentences for dummy PDF
SENTENCES = [
    "The results obtained from the experimental trials indicate a statistically significant correlation.",
    "Previous work in this domain has established a foundational understanding of the mechanisms involved.",
    "Data collected over a period of six months were analysed using standard regression techniques.",
    "The methodology adopted here builds upon the approach outlined by earlier investigators.",
    "These observations are consistent with the hypothesis that environmental factors mediate the response.",
    "It would appear that the relationship between the variables is non-linear in nature.",
    "The sample size, while modest, is sufficient for the purposes of this preliminary investigation.",
    "The findings provide empirical support for the theoretical framework presented earlier.",
    "Future research should address the limitations inherent in the current experimental design.",
    "A comprehensive review of the literature reveals a lack of consensus on this particular issue.",
    "The data were subjected to a rigorous quality control process prior to analysis.",
    "Statistical significance was determined using a two-tailed p-value threshold of 0.05.",
    "The observed effects were more pronounced in the treatment group than in the control group.",
    "The study highlights the importance of considering confounding variables in ecological models.",
    "Participants were selected using a randomized stratified sampling technique.",
    "The experimental apparatus was calibrated daily to ensure measurement accuracy.",
    "A significant body of evidence suggests that the phenomenon is more complex than previously thought.",
    "The results are interpreted within the context of existing socio-economic theories.",
    "Furthermore, the analysis indicates that the effect size is moderate across all tested conditions.",
    "In addition to the primary findings, several secondary patterns emerged from the data.",
    "The implications of these findings for public policy are discussed in the following section.",
    "No significant differences were observed between the various demographic subgroups.",
    "The robustness of the model was verified through multiple sensitivity analyses.",
    "Each trial was conducted in a controlled environment to minimize external interference.",
    "The authors acknowledge that the current study is limited by its retrospective nature.",
    "All procedures were approved by the institutional review board and followed ethical guidelines.",
    "The correlation coefficient indicates a strong positive relationship between X and Y.",
    "Overall, the research contributes to our understanding of the underlying biological processes."
]

def create_dummy_pdf(filename):
    doc = fitz.open()
    # Create 3 pages
    for page_num in range(3):
        page = doc.new_page()
        text = "\n\n".join(SENTENCES)
        page.insert_text((50, 50), text, fontsize=11)
    doc.save(filename)
    doc.close()
    print(f"Created dummy PDF: {filename}")

def test_pipeline():
    pdf_name = "dummy_academic.pdf"
    create_dummy_pdf(pdf_name)
    
    # Wait for server if needed (assumes it's running)
    base_url = "http://localhost:8001"
    
    print("\n--- Uploading PDF ---")
    with open(pdf_name, "rb") as f:
        resp = requests.post(f"{base_url}/train/upload", data={"field": "general"}, files={"file": f})
        print(f"Upload Status: {resp.status_code}")
        print(resp.json())
        
    print("\n--- Getting Quality Report ---")
    report_resp = requests.get(f"{base_url}/train/quality-report")
    print(f"Report Status: {report_resp.status_code}")
    print(json.dumps(report_resp.json(), indent=2))
    
    if os.path.exists(pdf_name):
        os.remove(pdf_name)

if __name__ == "__main__":
    test_pipeline()
