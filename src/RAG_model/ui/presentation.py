"""Shared visual language for the mounted Gradio research workspace."""

import gradio as gr

INK = "#101819"
SLATE = "#1A2628"
IVORY = "#F2EEE5"
TEAL = "#72D5BF"
COPPER = "#D7A477"
MUTED = "#A7B8B7"
BORDER = "#334649"


def create_research_theme() -> gr.themes.Base:
    """Keep the research palette consistent in both browser color modes."""
    theme = gr.themes.Base(
        primary_hue="teal",
        secondary_hue="teal",
        neutral_hue="slate",
        font=["Segoe UI", "system-ui", "sans-serif"],
        font_mono=["Consolas", "ui-monospace", "monospace"],
    )
    colors = {
        "body_background_fill": INK,
        "body_text_color": IVORY,
        "body_text_color_subdued": MUTED,
        "background_fill_primary": INK,
        "background_fill_secondary": SLATE,
        "border_color_primary": BORDER,
        "border_color_accent": TEAL,
        "border_color_accent_subdued": BORDER,
        "color_accent_soft": "#243F3B",
        "link_text_color": TEAL,
        "link_text_color_hover": "#A4E8D9",
        "link_text_color_active": TEAL,
        "link_text_color_visited": TEAL,
        "code_background_fill": "#203033",
        "block_background_fill": SLATE,
        "block_border_color": BORDER,
        "block_label_background_fill": SLATE,
        "block_label_border_color": BORDER,
        "block_label_text_color": MUTED,
        "block_title_background_fill": SLATE,
        "block_title_border_color": BORDER,
        "block_title_text_color": IVORY,
        "block_info_text_color": MUTED,
        "panel_background_fill": SLATE,
        "panel_border_color": BORDER,
        "accordion_text_color": IVORY,
        "input_background_fill": INK,
        "input_background_fill_focus": INK,
        "input_background_fill_hover": INK,
        "input_border_color": BORDER,
        "input_border_color_focus": TEAL,
        "input_border_color_hover": MUTED,
        "input_placeholder_color": MUTED,
        "loader_color": TEAL,
        "table_text_color": IVORY,
        "table_border_color": BORDER,
        "table_even_background_fill": SLATE,
        "table_odd_background_fill": INK,
        "table_row_focus": "#243F3B",
        "button_primary_background_fill": TEAL,
        "button_primary_background_fill_hover": "#A4E8D9",
        "button_primary_border_color": TEAL,
        "button_primary_border_color_hover": "#A4E8D9",
        "button_primary_text_color": INK,
        "button_primary_text_color_hover": INK,
        "button_secondary_background_fill": SLATE,
        "button_secondary_background_fill_hover": "#243638",
        "button_secondary_border_color": BORDER,
        "button_secondary_border_color_hover": TEAL,
        "button_secondary_text_color": IVORY,
        "button_secondary_text_color_hover": IVORY,
        "error_background_fill": "#382723",
        "error_border_color": COPPER,
        "error_text_color": IVORY,
        "error_icon_color": COPPER,
    }
    paired_colors = {
        variant: value
        for name, value in colors.items()
        for variant in (name, f"{name}_dark")
    }
    return theme.set(
        **paired_colors,
        color_accent=TEAL,
        block_radius="16px",
        block_border_width="1px",
        block_padding="20px",
        input_radius="12px",
        input_border_width="1px",
        button_large_radius="12px",
        button_large_text_weight="600",
        button_transition="background 160ms ease, border-color 160ms ease",
        layout_gap="20px",
        prose_text_size="15px",
        chatbot_text_size="15px",
    )


RESEARCH_HEADER = """
<header class="research-header">
  <div class="brand-line">
    <span class="brand-mark" aria-hidden="true">↗</span>
    <span>THE RESEARCH DESK</span>
    <span class="source-tag">SEC FILINGS</span>
  </div>
  <h1>Crypto Filing <span>Research.</span></h1>
  <p>Go beyond the ticker. Explore fund mechanics, compare disclosures,
  and follow the evidence back to the filing.</p>
  <div class="coverage-line">
    <div class="ticker-chips" aria-label="Supported tickers">
      <span>IBIT</span><span>ETHA</span><span>FBTC</span>
      <span>FETH</span><span>GBTC</span><span>ETHE</span>
    </div>
    <span class="coverage-note">Selected 10-K &amp; 10-Q filings · 2023–Q1 2026</span>
  </div>
</header>
"""

CHAT_PLACEHOLDER = """
<div class="chat-welcome">
  <span class="welcome-symbol" aria-hidden="true">↗</span>
  <h3>Start with a better question.</h3>
  <p>Explore how a fund works, trace a disclosure, or compare two tickers.</p>
  <p class="welcome-note">Answers use retrieved SEC evidence.<br>
  Supporting filings appear in the evidence panel.</p>
</div>
"""

# All component overrides are scoped to this app, using IDs/classes we own.
RESEARCH_CSS = """
.gradio-container:has(#research-app),
.gradio-container:has(#research-app) > .main { padding: 0 !important; }
#research-app {
    max-width: 1440px; margin: 0 auto; padding: 32px 40px 20px;
    background: #101819; color: #F2EEE5; color-scheme: dark;
}
#research-app .unframed-content .html-container { padding: 0; }
#research-app .research-header {
    padding: 8px 0 30px; border-bottom: 1px solid #334649;
}
#research-app .brand-line {
    display: flex; align-items: center; gap: 12px;
    font-size: 11px; letter-spacing: .18em; font-weight: 650; color: #A7B8B7;
}
#research-app .brand-mark {
    display: grid; place-items: center; width: 32px; height: 32px;
    border: 1px solid #72D5BF; border-radius: 10px; color: #72D5BF;
    font-size: 23px; letter-spacing: 0;
}
#research-app .source-tag {
    margin-left: auto; color: #D7A477; letter-spacing: .12em;
}
#research-app .research-header h1 {
    margin: 24px 0 12px; font-size: clamp(32px, 4vw, 52px);
    line-height: 1.08; letter-spacing: -.045em; font-weight: 650;
}
#research-app .research-header h1 span { color: #72D5BF; }
#research-app .research-header p {
    max-width: 650px; color: #A7B8B7; font-size: 16px; line-height: 1.65;
}
#research-app .coverage-line {
    display: flex; align-items: center; flex-wrap: wrap; gap: 16px; margin-top: 22px;
}
#research-app .ticker-chips { display: flex; flex-wrap: wrap; gap: 8px; }
#research-app .ticker-chips span {
    border: 1px solid #334649; border-radius: 6px; padding: 5px 10px;
    font-family: Consolas, monospace; font-size: 12px; color: #F2EEE5;
    background: #1A2628;
}
#research-app .coverage-note { color: #A7B8B7; font-size: 12px; }
#research-app #research-tabs { border: 0; }
#research-app #research-tabs [role="tablist"] {
    border-bottom: 1px solid #334649; gap: 6px; padding: 0 0 12px;
}
#research-app #research-tabs [role="tab"] {
    border-radius: 8px; border: 0; padding: 10px 20px; color: #A7B8B7;
}
#research-app #research-tabs [role="tab"][aria-selected="true"] {
    color: #72D5BF; background: #243F3B;
}
#research-app .tabitem { padding: 24px 0 0; border: 0; }
#research-app .workspace-heading {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 4px;
}
#research-app .eyebrow {
    color: #D7A477; font-size: 11px; letter-spacing: .14em; font-weight: 650;
}
#research-app .workspace-heading h2 { font-size: 19px; font-weight: 600; margin: 6px 0; }
#research-app .workspace-heading p { font-size: 12px; color: #A7B8B7; }
#research-app #conversation { background: #152022; }
#research-app #conversation .message { line-height: 1.75; }
#research-app #conversation .message.user { background: #243F3B; }
#research-app #conversation .message.bot { background: #1A2628; }
#research-app .chat-welcome { max-width: 390px; padding: 24px; }
#research-app .welcome-symbol { color: #72D5BF; font-size: 36px; }
#research-app .chat-welcome h3 { color: #F2EEE5; font-size: 23px; letter-spacing: -.025em; }
#research-app .chat-welcome p { color: #A7B8B7; line-height: 1.7; }
#research-app .chat-welcome .welcome-note { font-size: 12px; margin-top: 20px; }
#research-app #question-composer { align-items: stretch; gap: 12px; }
#research-app #question-input { padding: 14px 16px; }
#research-app #question-input textarea { line-height: 1.6; }
#research-app #ask-question { align-self: stretch; min-height: 84px; }
#research-app .example-card {
    text-align: left; justify-content: flex-start; white-space: normal;
    overflow-wrap: anywhere; line-height: 1.65; width: 100%;
    padding: 16px 20px; font-size: 14px; font-weight: 400; min-height: 0;
}
#research-app #example-prompts { gap: 10px; }
#research-app .example-card:hover { border-color: #72D5BF; }
#research-app #evidence-panel {
    background: #1A2628; border: 1px solid #334649; border-radius: 16px;
    padding: 22px; gap: 14px;
}
#research-app #citations { padding: 0; border: 0; background: transparent; }
#research-app #citations ul { list-style: none; padding: 0; }
#research-app #citations li {
    background: #101819; border: 1px solid #334649; border-left: 2px solid #D7A477;
    border-radius: 10px; padding: 16px; margin: 0 0 12px;
    font-size: 13px; line-height: 1.8;
}
#research-app #citations li strong { display: block; color: #F2EEE5; }
#research-app #citations li a { display: block; margin-top: 8px; }
#research-app .evidence-note { color: #A7B8B7; font-size: 12px; line-height: 1.7; }
#research-app #request-details { margin-top: 4px; }
#research-app #request-status { font-size: 12px; }
#research-app .section-intro { max-width: 900px; padding: 4px 0 12px; }
#research-app .section-intro h2 { font-size: 28px; letter-spacing: -.035em; }
#research-app .section-intro p { color: #A7B8B7; line-height: 1.75; }
#research-app .metric-chart { overflow: hidden; padding: 4px; }
#research-app .config-table { width: 100%; border-collapse: collapse; font-size: 13px; }
#research-app .config-table th, #research-app .config-table td {
    text-align: left; padding: 12px 16px; border-bottom: 1px solid #334649;
    overflow-wrap: anywhere;
}
#research-app .config-table th { color: #A7B8B7; font-weight: 500; }
#research-app .config-table td:first-child { width: 38%; }
#research-app #evaluation-prompt { scroll-margin-top: 24px; padding: 20px; }
#research-app #evaluation-prompt h3 { margin-bottom: 12px; font-size: 19px; }
#research-app #evaluation-prompt p { color: #A7B8B7; line-height: 1.7; margin-bottom: 16px; }
#research-app #evaluation-prompt pre {
    white-space: pre-wrap; overflow-wrap: anywhere; padding: 20px;
    background: #101819; border: 1px solid #334649; border-radius: 12px;
    font-size: 13px; line-height: 1.8;
}
#research-app .about-card { padding: 24px; }
#research-app .about-card h3 { font-size: 20px; margin-bottom: 16px; }
#research-app .about-card p, #research-app .about-card li { line-height: 1.8; }
#research-app .research-footer {
    border-top: 1px solid #334649; padding-top: 18px; margin-top: 8px;
    color: #A7B8B7; font-size: 12px;
    background: transparent; border-radius: 0;
}
#research-app a { text-underline-offset: 3px; }
#research-app :is(button, a, textarea, summary):focus-visible {
    outline: 2px solid #72D5BF; outline-offset: 3px;
}
#research-app :is(.prose, .message, #citations) { overflow-wrap: anywhere; }
#research-app :is(.prose, .message) table { display: block; overflow-x: auto; }
@media (max-width: 900px) {
    #research-app { padding: 24px 20px 16px; }
    #research-app #chat-workspace { flex-direction: column; }
    #research-app #chat-workspace > div { min-width: 0 !important; width: 100%; }
    #research-app #citations { height: auto !important; max-height: 360px; }
    #research-app .chart-row { flex-direction: column; }
    #research-app .chart-row > div { min-width: 0 !important; width: 100%; }
}
@media (max-width: 540px) {
    #research-app { padding: 18px 12px 12px; }
    #research-app .source-tag { font-size: 9px; }
    #research-app .brand-line { letter-spacing: .1em; font-size: 10px; gap: 8px; }
    #research-app #research-tabs [role="tab"] { padding: 10px 14px; }
    #research-app #conversation { height: 380px !important; }
    #research-app #question-composer, #research-app #example-prompts { flex-direction: column; }
    #research-app #question-composer > *, #research-app #example-prompts > * {
        min-width: 0 !important; width: 100%;
    }
    #research-app #ask-question { min-height: 48px; }
    #research-app .example-card { padding: 14px 16px; }
    #research-app #evidence-panel { padding: 18px; }
    #research-app #evaluation-prompt { padding: 12px; }
    #research-app #evaluation-prompt pre { padding: 12px; }
}
"""
