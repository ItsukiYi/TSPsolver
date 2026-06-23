"""Extract DualOpt experiment sections."""
import fitz, sys

sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open('DualOpt.pdf')
started = False
for i in range(5, min(15, len(doc))):
    page = doc[i]
    text = page.get_text()
    keywords = ['experiment', 'result', 'benchmark', 'table ', 'performance', 'evaluation',
                'dataset', 'random', 'tsplib', 'vlsi', 'compar', 'speedup']
    if any(kw in text.lower() for kw in keywords):
        if not started:
            print(f'=== EXPERIMENTS START AT PAGE {i+1} ===')
            started = True
        print(f'--- PAGE {i+1} ---')
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 15:
                words = line.split()
                if len(words) > 3:
                    # Remove null bytes
                    clean = line.replace('\x00', '')
                    if len(clean) > 10:
                        print(clean[:300])
        print()
doc.close()
