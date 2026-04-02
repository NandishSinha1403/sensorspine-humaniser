import os
import sys
import requests
import json
import argparse
from glob import glob

def build_corpus(directory, field="general"):
    base_url = "http://localhost:8001"
    
    # Check if trainer is up
    try:
        requests.get(f"{base_url}/train/health")
    except:
        print(f"Error: Training server not found at {base_url}. Run start.sh first.")
        sys.exit(1)

    # Find all PDFs recursively
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return

    print(f"🚀 Found {len(pdf_files)} PDFs. Starting corpus ingestion for field: {field}...")
    
    success_count = 0
    total_sentences = 0
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            with open(pdf_path, "rb") as f:
                resp = requests.post(
                    f"{base_url}/train/upload",
                    data={"field": field},
                    files={"file": f},
                    timeout=30
                )
                
            if resp.status_code == 200:
                data = resp.json()
                sent_count = data.get("sentences", 0)
                success_count += 1
                total_sentences += sent_count
                print(f"✓ {filename} — {sent_count} sentences extracted")
            else:
                print(f"✗ {filename} — skipped (Server error: {resp.status_code})")
        except Exception as e:
            print(f"✗ {filename} — skipped ({str(e)})")

    # Final Quality Report
    print("\n--- Final Quality Report ---")
    try:
        report_resp = requests.get(f"{base_url}/train/quality-report")
        if report_resp.status_code == 200:
            report = report_resp.json()
            # Find the specific field in report
            field_report = next((r for r in report if r["field"] == field), None)
            
            if field_report:
                print(f"Field: {field}")
                print(f"Average AI Score: {field_report['average']:.2f}")
                print(f"Verdict: {field_report['verdict'].upper()}")
                
                print("\n--- Summary ---")
                print(f"Total PDFs Processed: {success_count}/{len(pdf_files)}")
                print(f"Total Sentences in Corpus: {total_sentences}")
                print(f"Final Quality Rating: {'PASS' if field_report['average'] < 35 else 'REVIEW NEEDED'}")
            else:
                print("No report found for this field.")
    except Exception as e:
        print(f"Could not retrieve quality report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a style corpus from a folder of PDFs")
    parser.add_argument("directory", help="Path to folder containing PDFs")
    parser.add_argument("--field", default="general", help="Target field profile (default: general)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
        
    build_corpus(args.directory, args.field)
