"""
Stress test — run 10 diverse academic paragraphs through the humanizer
and dump original + humanized + scores to a readable markdown file.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.humanizer import humanize_text
from app.core.detector import detect_ai_score

PARAGRAPHS = [
    # 1. Climate Science (the user's original)
    """Climate change refers to long-term shifts in temperature and weather patterns driven largely by human activities, especially the burning of fossil fuels and land-use changes. Since the late 19th century, Earth's average surface temperature has risen by approximately 1.1 °C. This increase is primarily linked to higher concentrations of greenhouse gases such as carbon dioxide (CO₂), methane (CH₄), and nitrous oxide (N₂O), which trap heat in the atmosphere through the greenhouse effect. The impacts are widespread: more frequent and intense heatwaves, rising sea levels due to melting glaciers and thermal expansion of oceans, shifts in precipitation patterns, and increased occurrence of extreme weather events. These changes threaten ecosystems, biodiversity, food security, water resources, and human health. Scientific consensus, as reflected in reports by the Intergovernmental Panel on Climate Change (IPCC), emphasizes that limiting global warming to well below 2 °C above pre-industrial levels requires rapid reductions in emissions, a transition to renewable energy sources, and coordinated global policy efforts.""",

    # 2. Machine Learning (heavy AI-marker text)
    """Machine learning has become increasingly important in today's world, playing a crucial role in various domains. Furthermore, it is worth noting that deep learning, a subset of machine learning, has been shown to outperform traditional methods in a wide range of applications. Additionally, research has shown that neural networks can effectively process large datasets and extract meaningful patterns. In conclusion, the development of AI systems has significant implications for society. It is essential to consider the ethical dimensions of these technologies. The impact of artificial intelligence on healthcare, education, and transportation is undeniable. Moreover, one must consider that responsible development requires a combination of technical expertise and ethical awareness.""",

    # 3. Quantum Computing
    """Quantum computing represents a fundamentally different approach to information processing that leverages the principles of quantum mechanics. Unlike classical computers that process information in binary bits, quantum computers use quantum bits, or qubits, which can exist in superpositions of states. This property, along with entanglement and quantum interference, enables quantum computers to solve certain classes of problems exponentially faster than their classical counterparts. Notable algorithms include Shor's algorithm for integer factorization and Grover's algorithm for unstructured search. However, significant challenges remain, including qubit decoherence, error correction, and the need for extremely low operating temperatures. Current quantum processors from companies like IBM, Google, and IonQ have demonstrated quantum advantage in specific benchmarks, though practical large-scale quantum computing remains years away.""",

    # 4. Psychology / Cognitive Science
    """Cognitive behavioral therapy (CBT) is one of the most widely studied and empirically supported forms of psychotherapy. It is based on the premise that psychological distress is maintained by dysfunctional thought patterns and maladaptive behaviors. Furthermore, CBT emphasizes the importance of identifying and challenging cognitive distortions, such as catastrophizing, overgeneralization, and all-or-nothing thinking. Additionally, it is worth noting that numerous randomized controlled trials have demonstrated the efficacy of CBT for conditions including depression, anxiety disorders, post-traumatic stress disorder, and obsessive-compulsive disorder. The therapeutic process typically involves collaborative goal-setting, homework assignments, and the development of coping strategies. In recent years, CBT has been adapted for delivery through digital platforms, making it more accessible to underserved populations.""",

    # 5. Economics (formal academic)
    """The relationship between monetary policy and inflation has been a central topic in macroeconomic theory since the work of Milton Friedman and the monetarist school. Central banks, such as the Federal Reserve and the European Central Bank, use interest rate adjustments, open market operations, and forward guidance to influence money supply and credit conditions. When inflation exceeds target levels, contractionary monetary policy — raising interest rates and reducing bond purchases — is typically employed to cool aggregate demand. Conversely, during periods of deflation or recession, expansionary policies are implemented to stimulate economic activity. The Taylor Rule provides a prescriptive framework for setting the federal funds rate based on deviations of inflation from its target and output from its potential. Recent debates have focused on the effectiveness of unconventional monetary policies, such as quantitative easing and negative interest rates, particularly in the context of the post-2008 financial crisis and the COVID-19 pandemic.""",

    # 6. Neuroscience
    """The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections that constitute the neural substrate of cognition, emotion, and behavior. Neurotransmitters such as dopamine, serotonin, and glutamate mediate signal transmission across synaptic clefts, and imbalances in these chemical messengers have been implicated in conditions ranging from Parkinson's disease to major depressive disorder. Neuroplasticity, the brain's capacity to reorganize its structure and function in response to experience, underpins learning and memory formation. Advances in neuroimaging techniques, including functional magnetic resonance imaging (fMRI) and diffusion tensor imaging (DTI), have enabled researchers to map neural circuits with unprecedented spatial resolution. The Human Brain Project and the BRAIN Initiative represent large-scale collaborative efforts to decode the connectome and develop computational models of brain function.""",

    # 7. Environmental Science (AI-heavy transitions)
    """Biodiversity loss is a subject of growing concern in the scientific community. It is important to note that the current rate of species extinction is estimated to be 100 to 1,000 times higher than the natural background rate. Furthermore, habitat destruction, driven by agricultural expansion, urbanization, and deforestation, plays a crucial role in this crisis. Additionally, invasive species, pollution, and climate change are some of the factors contributing to ecosystem degradation. In light of this, conservation strategies must address multiple drivers simultaneously. The development of protected areas, wildlife corridors, and habitat restoration programs is essential for preserving biodiversity. Moreover, community-based conservation approaches have been shown to be effective in regions where local livelihoods depend on natural resources.""",

    # 8. Computer Science / Networking
    """Software-defined networking (SDN) decouples the control plane from the data plane in network architecture, enabling centralized programmability and dynamic traffic management. The OpenFlow protocol serves as the primary communication standard between the SDN controller and network switches. This architectural paradigm offers several advantages over traditional distributed networking, including simplified network configuration, improved resource utilization, and enhanced security policy enforcement. In enterprise environments, SDN facilitates rapid service provisioning and enables network virtualization through technologies such as VXLAN and NVGRE. However, the centralized controller model introduces potential single points of failure, necessitating redundancy mechanisms and distributed controller architectures such as ONOS and OpenDaylight. Recent research has explored the integration of machine learning algorithms with SDN controllers for intelligent traffic classification and anomaly detection.""",

    # 9. Philosophy (dense academic prose)
    """The problem of consciousness remains one of the most intractable questions in philosophy of mind. David Chalmers' formulation of the "hard problem" — explaining why and how physical processes give rise to subjective experience — has generated extensive debate since its introduction in 1995. Physicalist accounts, including identity theory and functionalism, maintain that consciousness is reducible to or supervenes upon neural processes. Dualist perspectives, by contrast, argue that phenomenal experience possesses properties that cannot be captured by any purely physical description. More recently, integrated information theory (IIT), proposed by Giulio Tononi, attempts to quantify consciousness through a mathematical measure called phi, which captures the degree of integrated information within a system. Panpsychism, once dismissed as a fringe position, has gained renewed philosophical attention through the work of Philip Goff and Galen Strawson.""",

    # 10. Public Health (mix of AI markers + technical)
    """The COVID-19 pandemic has highlighted the critical importance of global health infrastructure and pandemic preparedness. It is essential to consider that vaccine development, which traditionally takes 10 to 15 years, was accelerated to under a year through unprecedented international collaboration and mRNA technology platforms. Furthermore, the pandemic exposed significant disparities in healthcare access between high-income and low-income countries. Additionally, the role of public health communication in combating misinformation has become increasingly apparent. Research has shown that vaccine hesitancy is driven by a combination of mistrust in institutions, exposure to conspiracy theories, and insufficient health literacy. The development of robust surveillance systems, equitable vaccine distribution mechanisms, and coordinated international response protocols remains crucial for addressing future pandemics.""",
]

LABELS = [
    "Climate Science",
    "Machine Learning (AI-heavy)",
    "Quantum Computing",
    "Psychology / CBT",
    "Economics / Monetary Policy",
    "Neuroscience",
    "Environmental Science (AI-heavy)",
    "Computer Science / SDN",
    "Philosophy of Mind",
    "Public Health / COVID-19",
]

def run_stress_test():
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_test_results.md")
    
    with open(output_path, "w") as f:
        f.write("# ScholarAI Pipeline Stress Test Results\n\n")
        f.write("---\n\n")
        
        for idx, (para, label) in enumerate(zip(PARAGRAPHS, LABELS), 1):
            print(f"[{idx}/10] Processing: {label}...")
            result = humanize_text(para, intensity=0.8)
            
            f.write(f"## Test {idx}: {label}\n\n")
            f.write(f"**Original Score:** {result['original_score']:.1f} → **Humanized Score:** {result['humanized_score']:.1f}\n\n")
            
            # Changes summary
            active_changes = {k: v for k, v in result['changes_made'].items() if v > 0}
            if active_changes:
                f.write("**Changes:** " + ", ".join(f"{k}={v}" for k, v in sorted(active_changes.items())) + "\n\n")
            
            f.write("### Original\n")
            f.write(f"> {para.strip()}\n\n")
            
            f.write("### Humanized\n")
            f.write(f"> {result['humanized_text'].strip()}\n\n")
            
            # Flag potential issues
            issues = []
            htext = result['humanized_text']
            
            # Check for sentence fragments (sentences < 4 words ending with period)
            import re
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(htext)
            for s in sents:
                words = s.strip().split()
                if len(words) < 3 and len(words) > 0:
                    issues.append(f"⚠️ Very short fragment: '{s.strip()}'")
            
            # Check for broken sentences (period followed by lowercase)
            broken = re.findall(r'\.\s+[a-z]', htext)
            if broken:
                for b in broken:
                    context_idx = htext.find(b)
                    context = htext[max(0,context_idx-20):context_idx+30]
                    issues.append(f"⚠️ Lowercase after period: '...{context}...'")
            
            # Check for double articles
            doubles = re.findall(r'\b(a|an|the)\s+(a|an|the)\s+', htext, re.IGNORECASE)
            if doubles:
                issues.append(f"⚠️ Double articles found: {doubles}")
            
            # Check for known bad replacements
            bad_words = ["throttles", "floors", "normals", "componentry", "leveraging"]
            for bw in bad_words:
                if bw in htext.lower():
                    issues.append(f"⚠️ Bad synonym: '{bw}' found in output")
            
            # Check for colloquial parentheticals in academic text
            colloquial = ["if you will", "so to speak", "quite literally", "to be fair", "for better or worse"]
            for c in colloquial:
                if c in htext.lower():
                    issues.append(f"⚠️ Colloquial phrase: '{c}' in academic text")
            
            # Check for duplicate connectors
            dup_conn = re.findall(r'\b(however|moreover|furthermore|nevertheless|consequently)\s*,?\s*(however|moreover|furthermore|nevertheless|consequently)', htext, re.IGNORECASE)
            if dup_conn:
                issues.append(f"⚠️ Duplicate connectors: {dup_conn}")
            
            # Check for "thoroughly contextualized"
            if "thoroughly contextualized" in htext.lower():
                issues.append("⚠️ Bad jargon injection: 'thoroughly contextualized'")
            
            if issues:
                f.write("### ⚠️ Issues Detected\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
            else:
                f.write("### ✅ No issues detected\n\n")
            
            f.write("---\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| # | Category | Original | Humanized | Issues |\n")
        f.write("|---|----------|----------|-----------|--------|\n")
        
        # Re-run for summary (we already have results but let's just format)
        for idx, (para, label) in enumerate(zip(PARAGRAPHS, LABELS), 1):
            result = humanize_text(para, intensity=0.8)
            htext = result['humanized_text']
            issue_count = 0
            sents = sent_tokenize(htext)
            for s in sents:
                if len(s.strip().split()) < 3:
                    issue_count += 1
            if re.findall(r'\b(a|an|the)\s+(a|an|the)\s+', htext, re.IGNORECASE):
                issue_count += 1
            for bw in ["throttles", "floors", "normals", "componentry", "leveraging"]:
                if bw in htext.lower():
                    issue_count += 1
            status = "✅" if issue_count == 0 else f"⚠️ {issue_count}"
            f.write(f"| {idx} | {label} | {result['original_score']:.0f} | {result['humanized_score']:.0f} | {status} |\n")
    
    print(f"\n✅ Results saved to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    run_stress_test()
