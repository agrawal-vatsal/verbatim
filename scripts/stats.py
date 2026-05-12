from verbatim.db import Database


def display_stats() -> None:
    db = Database()
    s = db.get_system_stats()
    l = s['latency']
    d = s['distance']

    print("\n" + "═" * 60)
    print("📊 VERBATIM OBSERVABILITY DASHBOARD".center(60))
    print("═" * 60)

    # Latency Table
    print(f"\n⏳ LATENCY PIPELINE (24H)")
    print(f"   {'Phase':<15} | {'Average':>12} | {'P95 (Tail)':>12}")
    print(f"   {'-' * 45}")
    print(f"   {'🧠 Processing':<15} | {l['avg']['p']:>10.0f}ms | {l['p95']['p']:>10.0f}ms")
    print(f"   {'🔍 Retrieval':<15} | {l['avg']['r']:>10.0f}ms | {l['p95']['r']:>10.0f}ms")
    print(f"   {'🔀 Reranking':<15} | {l['avg']['rr']:>10.0f}ms | {l['p95']['rr']:>10.0f}ms")
    print(f"   {'✍️ Synthesis':<15} | {l['avg']['s']:>10.0f}ms | {l['p95']['s']:>10.0f}ms")
    print(f"   {'-' * 45}")
    print(f"   {'🚀 TOTAL E2E':<15} | {l['avg']['t']:>10.0f}ms | {l['p95']['t']:>10.0f}ms")

    # RAG Quality
    print(f"\n🎯 RAG QUALITY (DISTANCE)")
    status_p50 = "🟢" if d['p50'] < 0.4 else "🟡"
    status_p95 = "🟢" if d['p95'] < 0.5 else "🔴"
    print(f"   • Median (P50):  {d['p50']:.4f} {status_p50}")
    print(f"   • Worst  (P95):  {d['p95']:.4f} {status_p95} (Lower is better)")

    # Retrieval Signal Breakdown
    sig = s['retrieval_signal']
    total_sig = sig['overlap'] + sig['vector_only'] + sig['keyword_only']
    def pct(v: float) -> str:
        return f"{v / total_sig * 100:.0f}%" if total_sig > 0 else "N/A"

    print(f"\n🔀 RETRIEVAL SIGNAL (AVG CHUNKS PER QUERY)")
    print(f"   • Both engines:  {sig['overlap']:.1f}  ({pct(sig['overlap'])})")
    print(f"   • Vector only:   {sig['vector_only']:.1f}  ({pct(sig['vector_only'])})")
    print(f"   • Keyword only:  {sig['keyword_only']:.1f}  ({pct(sig['keyword_only'])})")

    # Reranker Displacement
    disp = s['displacement']
    disp_p50_status = "🟢" if disp['p50'] > 0.1 else "🟡"
    print(f"\n🔁 RERANKER DISPLACEMENT (0=no change, 1=full reorder)")
    print(f"   • Median (P50):  {disp['p50']:.4f} {disp_p50_status}")
    print(f"   • Tail   (P95):  {disp['p95']:.4f}")

    # Storage
    print(f"\n🏢 CONTENT COVERAGE")
    for company, count in s['company_dist']:
        print(f"   • {company[:20]:<20} {count:>8} chunks")

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    display_stats()
