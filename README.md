# Baseline for Ambient AI Bull + Bear Debate + Summarizer with main3.py

## Highlights
1. Built end-to-end on Pathway: streaming file source, reducers, UDFs, grouping, and dedup logic. 
2. Ambient AI design: edits to your markdown reports automatically trigger a controlled LLM workflow (Bull vs Bear rounds + summarizer) so insights arrive without manual orchestration.
3. Smart change detection: **content hashing** + dedup prevents repeated runs on identical inputs: saves tokens and keeps history clean.
4. Safe and modular LLM usage: model calls are isolated to UDFs, easy to swap models or add retries/backoff.
5. Production touches: atomic writes, persistent dedupe state, and easy hooks for logging/metrics and per-group deduplication.

# main3.py
Drop fundamentals.md, market_research.md, news.md, sentiment.md into data-source/.
Set OPENAI_API_KEY and run the pipeline (python main3.py).
Edit files — the ambient system detects changes, runs the debate, and emits final_summary_*.txt + debate_results.jsonlines

**main1.py** does not use pathway, while **main2.py** uses time check for ambient orchestration which does not fully utilize powers of pathway. 
