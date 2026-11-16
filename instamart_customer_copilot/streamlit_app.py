"""
Instamart Customer Intelligence & Support Copilot
Run from this directory: streamlit run streamlit_app.py
"""
from pathlib import Path
import sqlite3
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLEAN = DATA / "clean"
OUTPUTS = ROOT / "outputs"
DB = DATA / "instamart.db"

st.set_page_config(
    page_title="Instamart Copilot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e8edf4;
        padding: 15px 16px; border-radius: 12px;
    }
    div[data-testid="stSidebar"] {border-right: 1px solid #e8edf4;}
    .hero {
        padding: 22px 24px; border-radius: 16px;
        background: linear-gradient(110deg, #102a43, #176b62);
        color: white; margin-bottom: 18px;
    }
    .hero h1 {color: white; margin: 0 0 6px 0;}
    .hero p {color: #e2f2ef; margin: 0;}
    .small-note {color: #64748b; font-size: 0.88rem;}
</style>
""", unsafe_allow_html=True)


# ---------- Data access ----------
@st.cache_data(show_spinner=False)
def load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def load_table(table: str):
    if not DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)

@st.cache_data(show_spinner=False)
def load_refund_ledger():
    path = OUTPUTS / "refund_ledger.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "tool", "order_id", "refund_inr"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)

def money(value):
    return f"₹{value:,.2f}"

orders = load_table("orders")
customers = load_table("customers")
products = load_table("products")
segments = load_csv(str(OUTPUTS / "customer_segments.csv"))
retention = load_csv(str(OUTPUTS / "retention_list.csv"))
tickets = load_csv(str(OUTPUTS / "triage_results.csv"))

if not orders.empty:
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🛒 Instamart Copilot")
    st.caption("Customer intelligence • Support • AI")
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Order Explorer",
            "Customer Intelligence",
            "Support Tickets",
            "Policy Assistant",
            "Refund Center",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("Project workspace")
    st.write("🟢 Local data mode")
    st.caption("Synthetic learning-project data. No live customer system is connected.")

st.markdown(
    '<div class="hero"><h1>Instamart Customer Copilot</h1>'
    '<p>One workspace to understand orders, customers, support requests and policy decisions.</p></div>',
    unsafe_allow_html=True,
)

if orders.empty:
    st.error("Order data is missing. Confirm that data/instamart.db exists and run your data pipeline.")
    st.stop()


# ---------- Overview ----------
if page == "Overview":
    st.subheader("Business overview")
    st.caption("Use the filters to explore your synthetic Instamart dataset.")

    date_min = orders["order_ts"].min().date()
    date_max = orders["order_ts"].max().date()
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        start_date = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    with c2:
        end_date = st.date_input("To", value=date_max, min_value=date_min, max_value=date_max)
    with c3:
        city_options = ["All cities"] + sorted(customers["city"].dropna().unique().tolist())
        selected_city = st.selectbox("City", city_options)

    filtered = orders[
        orders["order_ts"].dt.date.between(start_date, end_date)
    ].copy()
    if start_date > end_date:
        st.warning("The start date must be on or before the end date.")
        st.stop()
    if selected_city != "All cities":
        city_customers = set(customers.loc[customers.city == selected_city, "customer_id"])
        filtered = filtered[filtered.customer_id.isin(city_customers)]

    completed = filtered[filtered.status == "completed"]
    revenue = completed["order_amount"].sum()
    avg_order = completed["order_amount"].mean() if len(completed) else 0
    delivery_avg = completed["delivery_minutes"].mean() if len(completed) else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total orders", f"{len(filtered):,}")
    m2.metric("Completed revenue", money(revenue))
    m3.metric("Unique customers", f"{filtered.customer_id.nunique():,}")
    m4.metric("Average delivery", f"{delivery_avg:.1f} min" if pd.notna(delivery_avg) else "—")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Revenue over time")
        daily = completed.assign(day=completed.order_ts.dt.date).groupby("day", as_index=False).order_amount.sum()
        if not daily.empty:
            fig = px.line(daily, x="day", y="order_amount", markers=True,
                          labels={"day": "Order date", "order_amount": "Revenue (₹)"})
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed orders in this date range.")
    with right:
        st.markdown("#### Order status")
        status_df = filtered.status.value_counts().rename_axis("status").reset_index(name="orders")
        fig = px.pie(status_df, names="status", values="orders", hole=0.55)
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=330)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Category performance")
    if not products.empty:
        cat = completed.merge(products[["product_id", "category"]], on="product_id", how="left")
        cat = cat.groupby("category", as_index=False).order_amount.sum().sort_values("order_amount", ascending=False)
        fig = px.bar(cat, x="category", y="order_amount", text_auto=".2s",
                     labels={"category": "Category", "order_amount": "Revenue (₹)"})
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=330)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View filtered order records"):
        st.dataframe(filtered.sort_values("order_ts", ascending=False), use_container_width=True, hide_index=True)


# ---------- Order explorer ----------
elif page == "Order Explorer":
    st.subheader("Order explorer")
    st.caption("Search by order/customer ID and narrow results using status and delivery filters.")
    q = st.text_input("Search order ID or customer ID", placeholder="e.g. ORD-100001 or C0001")
    status_values = ["All"] + sorted(orders.status.dropna().unique().tolist())
    f1, f2 = st.columns(2)
    with f1:
        status_filter = st.selectbox("Order status", status_values)
    with f2:
        slow_only = st.checkbox("Show deliveries over 45 minutes only")
    view = orders.copy()
    if q.strip():
        view = view[
            view.order_id.astype(str).str.contains(q.strip(), case=False, na=False)
            | view.customer_id.astype(str).str.contains(q.strip(), case=False, na=False)
        ]
    if status_filter != "All":
        view = view[view.status == status_filter]
    if slow_only:
        view = view[view.delivery_minutes > 45]
    st.write(f"**{len(view):,} orders found**")
    st.dataframe(view.sort_values("order_ts", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download filtered orders CSV", view.to_csv(index=False).encode("utf-8"),
                       "instamart_filtered_orders.csv", "text/csv")


# ---------- Customer intelligence ----------
elif page == "Customer Intelligence":
    st.subheader("Customer intelligence")
    if segments.empty:
        st.warning("Customer segment output is missing. Run the Part B segmentation pipeline first.")
        st.stop()
    s1, s2, s3 = st.columns(3)
    s1.metric("Segmented customers", f"{segments.customer_id.nunique():,}")
    s2.metric("Premium customers", f"{int(segments.is_premium.astype(str).str.lower().eq('true').sum()):,}")
    s3.metric("Average customer spend", money(segments.monetary.mean()))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Segment distribution")
        counts = segments.segment_name.value_counts().rename_axis("segment").reset_index(name="customers")
        fig = px.bar(counts, x="segment", y="customers", color="segment",
                     labels={"segment": "Segment", "customers": "Customers"})
        fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Recency vs monetary value")
        fig = px.scatter(segments, x="recency", y="monetary", color="segment_name",
                         hover_data=["customer_id", "frequency"],
                         labels={"recency": "Days since last order", "monetary": "Total spend (₹)"})
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explore customers")
    a, b = st.columns(2)
    with a:
        segment_filter = st.multiselect("Customer segment", sorted(segments.segment_name.dropna().unique()),
                                        default=sorted(segments.segment_name.dropna().unique()))
    with b:
        min_spend = st.number_input("Minimum spend (₹)", min_value=0.0, value=0.0, step=100.0)
    customer_view = segments[
        segments.segment_name.isin(segment_filter) & (segments.monetary >= min_spend)
    ].merge(customers, on="customer_id", how="left")
    st.dataframe(customer_view.sort_values("monetary", ascending=False),
                 use_container_width=True, hide_index=True)
    st.download_button("Download customer segment CSV", customer_view.to_csv(index=False).encode("utf-8"),
                       "customer_intelligence.csv", "text/csv")

    if not retention.empty:
        st.markdown("#### Retention action list")
        st.caption("Customers identified by the lapse model for retention follow-up.")
        st.dataframe(retention, use_container_width=True, hide_index=True)


# ---------- Support tickets ----------
elif page == "Support Tickets":
    st.subheader("Support ticket center")
    if tickets.empty:
        st.warning("Ticket results are missing. Run the Part C triage pipeline first.")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Total tickets", f"{len(tickets):,}")
    b.metric("High priority", f"{int(tickets.priority.eq('HIGH').sum()):,}")
    c.metric("Human review", f"{int(tickets.route.eq('human_review').sum()):,}")

    x, y, z = st.columns(3)
    with x:
        category = st.multiselect("Category", sorted(tickets.category.dropna().unique()),
                                  default=sorted(tickets.category.dropna().unique()))
    with y:
        priority = st.multiselect("Priority", sorted(tickets.priority.dropna().unique()),
                                  default=sorted(tickets.priority.dropna().unique()))
    with z:
        route = st.multiselect("Routing queue", sorted(tickets.route.dropna().unique()),
                               default=sorted(tickets.route.dropna().unique()))
    ticket_view = tickets[
        tickets.category.isin(category) & tickets.priority.isin(priority) & tickets.route.isin(route)
    ]
    st.write(f"**{len(ticket_view)} matching tickets**")
    st.dataframe(ticket_view, use_container_width=True, hide_index=True)
    st.download_button("Download ticket report", ticket_view.to_csv(index=False).encode("utf-8"),
                       "ticket_triage_report.csv", "text/csv")
    st.markdown("#### Ticket category breakdown")
    chart = tickets.category.fillna("Unclassified").value_counts().rename_axis("category").reset_index(name="count")
    fig = px.pie(chart, names="category", values="count", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)


# ---------- Policy assistant ----------
elif page == "Policy Assistant":
    st.subheader("AI policy assistant")
    st.caption("Ask questions about the local policy documents. Answers are generated by the project's offline mock LLM.")
    question = st.text_area("Your policy question", placeholder="Example: What happens if a completed order is delivered more than 45 minutes late?")
    top_k = st.slider("Number of policy passages to retrieve", min_value=1, max_value=5, value=3)
    if st.button("Search policies", type="primary", disabled=not question.strip()):
        try:
            from src.policy_rag import PolicyRAG
            with st.spinner("Searching policy documents and preparing an answer..."):
                rag = PolicyRAG()
                hits = rag.retrieve(question, k=top_k)
                context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
                import mock_llm
                response = mock_llm.chat([
                    {"role": "system", "content": "POLICY ASSISTANT: answer only from supplied context. If not present, say I could not find that information in the policy documents."},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}
                ])
                answer = response.choices[0].message.content
            st.markdown("#### Answer")
            st.success(answer)
            st.markdown("#### Retrieved policy passages")
            for i, hit in enumerate(hits, 1):
                with st.expander(f"{i}. {hit['source']} • relevance {hit['score']:.3f}"):
                    st.write(hit["text"])
        except Exception as exc:
            st.error(f"Policy assistant could not run: {exc}")
            st.info("Check that mock_llm.py and the files in data/policies/ are present.")


# ---------- Refund center ----------
elif page == "Refund Center":
    st.subheader("Order refund center")
    st.caption("This is a simulator for the project's refund policy. It does not contact a real payment system.")
    order_id = st.text_input("Enter order ID", placeholder="e.g. ORD-100001").strip().upper()
    if order_id:
        match = orders[orders.order_id.astype(str).str.upper() == order_id]
        if match.empty:
            st.error("Order ID not found.")
        else:
            order = match.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Order status", str(order.status).title())
            c2.metric("Order amount", money(float(order.order_amount)))
            c3.metric("Delivery time", f"{order.delivery_minutes:g} min" if pd.notna(order.delivery_minutes) else "Not available")
            if order.status != "completed":
                st.warning("This order is not completed. It is not eligible for a refund under the current project rules.")
            elif pd.isna(order.delivery_minutes):
                st.warning("Delivery time is missing. Human review is required.")
            elif float(order.delivery_minutes) <= 45:
                st.info("No refund: delivery time is 45 minutes or less.")
            else:
                amount = float(order.order_amount)
                if amount > 1000:
                    st.warning(f"Potential refund: {money(amount)}. This exceeds ₹1,000 and must be escalated for human approval.")
                    action = "escalate_to_human"
                else:
                    st.success(f"Eligible for a simulated full refund of {money(amount)}.")
                    action = "issue_refund"
                confirm = st.checkbox("I understand this only records a simulated decision in the local refund ledger.")
                if st.button("Record refund decision", type="primary", disabled=not confirm):
                    OUTPUTS.mkdir(exist_ok=True)
                    entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tool": action,
                        "order_id": order_id,
                        "refund_inr": round(amount, 2),
                    }
                    with (OUTPUTS / "refund_ledger.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                    load_refund_ledger.clear()
                    st.success("Simulated decision recorded in the local refund ledger.")
                    st.rerun()

    st.markdown("#### Recent refund decisions")
    ledger = load_refund_ledger()
    if ledger.empty:
        st.caption("No refund decisions have been recorded yet.")
    else:
        st.dataframe(ledger.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download refund ledger", ledger.to_csv(index=False).encode("utf-8"),
                           "refund_ledger.csv", "text/csv")

st.divider()
st.markdown('<p class="small-note">Instamart Customer Copilot • Educational demo • All project data is synthetic</p>',
            unsafe_allow_html=True)
