"""Builds the printable report (.docx) and the 5-slide deck (.pptx) from out/results.json."""
import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Pt, Inches, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PRGB
from pptx.util import Inches as PI, Pt as PPt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
R = json.load(open(os.path.join(OUT, "results.json")))
E = R["evaluation"]
STUDENTS = [("Moosa Kaleemullah", "100523733076"), ("Siddharth Wali", "100523733061")]
TITLE = "Project 7: Self-Driving Lab Simulator"
SUB = "Reinforcement learning for choosing reaction temperature and concentration to maximise yield"


def fmt(k):
    e = E[k]
    return (f"{e['final_yield_mean']:.1f} ± {e['final_yield_std']:.1f}",
            f"{e['return_mean']:.2f} ± {e['return_std']:.2f}",
            f"{e['pct_reaching_global_opt']:.0f}%")


# =============================================================================
# REPORT
# =============================================================================
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9)
    s.top_margin = s.bottom_margin = Inches(0.8)


def para(text, bold=False, size=None, align=None, space_after=6, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align is not None: p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text, boldlead=None):
    p = doc.add_paragraph(style="List Bullet")
    if boldlead:
        p.add_run(boldlead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)
    return p


def table(rows, header=True, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""
            run = c.paragraphs[0].add_run(str(v)); run.font.size = Pt(10)
            if header and i == 0: run.bold = True
            if widths: c.width = Inches(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ---------------- Front page ----------------
for _ in range(3): doc.add_paragraph()
para("Reinforcement Learning — Unit 5 Homework Project", size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
para(TITLE, bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER)
para(SUB, italic=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
para("Category A: RL for Scientific Discovery — Active learning for synthesizing new materials",
     size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
para("Submitted by", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
t = doc.add_table(rows=len(STUDENTS) + 1, cols=2); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, row in enumerate([("Name", "Roll No.")] + STUDENTS):
    for j, v in enumerate(row):
        c = t.cell(i, j); c.text = ""; c.width = Inches(2.6)
        r = c.paragraphs[0].add_run(v); r.font.size = Pt(13); r.bold = (i == 0)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()
para("Deliverables: (1) Problem formulation  (2) Q-learning implementation  "
     "(3) Learning curves vs. baselines  (4) 5-slide presentation",
     size=10, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
doc.add_page_break()

# ---------------- 1. Problem formulation (one page) ----------------
doc.add_heading("1. Problem Formulation", level=1)
para("Goal. A self-driving (autonomous) laboratory must find the reaction conditions that give the "
     "highest product yield while spending as few experiments as possible. Each experiment is expensive "
     "and its measured yield is noisy. We model one experimental campaign as an episode of a Markov "
     "Decision Process and train an agent to choose the next recipe.")
doc.add_heading("Environment (simulated reaction)", level=2)
para("The hidden yield surface Y(T, C) is unknown to the agent. It has a global optimum near "
     f"T = {R['global_optimum']['T']:.0f} °C, C = {R['global_optimum']['C']:.1f} mol/L "
     f"(≈{R['global_optimum']['yield']:.0f}% yield), a decoy local optimum near 40 °C / 0.5 M (≈50% yield), "
     "and product decomposition above 100 °C. Every measurement adds Gaussian noise (σ = 3 percentage points). "
     "A campaign has a budget of 30 experiments and starts from a random recipe.")
table([
    ["Element", "Definition"],
    ["State s", "Current recipe (temperature index, concentration index) on an 11 × 11 grid: "
                "T ∈ {20, 30, …, 120} °C, C ∈ {0.1, 0.3, …, 2.1} mol/L → 121 states"],
    ["Action a", "5 discrete actions: T +10 °C, T −10 °C, C +0.2 M, C −0.2 M, repeat current recipe"],
    ["Transition", "Deterministic move on the grid (clipped at limits); the new recipe is then run"],
    ["Reward r", "r = measured_yield / 100 − 0.05 · max(0, (T − 80) / 40)  "
                 "(yield gained minus an energy/safety cost for hot runs)"],
    ["Episode", "30 experiments (budget). Return = total (cost-adjusted) yield over the campaign"],
    ["Discount γ", "0.97"],
], widths=[1.3, 5.3])
doc.add_heading("Why RL (and not just optimisation)?", level=2)
para("The return rewards every experiment, not just the last one, so the agent must trade off exploration "
     "(moving to new conditions) against exploitation (repeating a good recipe). This is exactly the "
     "budget-constrained decision problem a real automated lab faces.")
doc.add_page_break()

# ---------------- 2. Implementation ----------------
doc.add_heading("2. Implementation", level=1)
para("Algorithm: tabular Q-learning (off-policy temporal-difference control), implemented from scratch in "
     "Python/NumPy (file lab_rl.py, full listing in the Appendix).")
para("Q(s,a) ← Q(s,a) + α [ r + γ · maxₐ′ Q(s′,a′) − Q(s,a) ]", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
table([
    ["Hyper-parameter", "Value"],
    ["Step size α", "α = max(0.02, 0.5/√N(s,a)) — decays with visit count to average out measurement noise"],
    ["Discount γ", "0.97"],
    ["Exploration", "ε-greedy, ε decays linearly 1.0 → 0.05 over 1500 episodes"],
    ["Training", "3000 episodes (campaigns) × 30 steps, 5 independent random seeds"],
    ["Evaluation", "Greedy policy, 200 fresh campaigns per seed (1000 total)"],
], widths=[2.0, 4.6])
doc.add_heading("Baselines", level=2)
bullet(" picks one of the 5 actions uniformly at random every step.", "Random policy —")
bullet(" what a chemist might do by hand: try a neighbouring recipe; if yield improved, keep it, "
       "otherwise go back and try another direction; when no neighbour helps, keep repeating the best recipe.",
       "Hill-climbing heuristic —")
doc.add_heading("Metrics", level=2)
bullet(" sum of rewards over the 30-run campaign (what the agent optimises).", "Episode return:")
bullet(" mean true (noise-free) yield of the last 5 runs, i.e. the recipe the lab converged to.",
       "Converged yield:")
bullet(" fraction of campaigns whose converged yield ≥ 75% (the global peak region).", "Success rate:")

# ---------------- 3. Results ----------------
doc.add_heading("3. Results: Learning Curves vs. Baselines", level=1)
doc.add_picture(os.path.join(OUT, "learning_curves.png"), width=Inches(6.6))
para("Figure 1. Learning curves (mean ± std over 5 seeds, 25-episode moving average). "
     "Baselines do not learn, so their curves are flat.", italic=True, size=9)
table([
    ["Method", "Converged yield (%)", "Episode return", "Reached global optimum"],
    ["Random policy", *fmt("random")],
    ["Hill-climbing heuristic", *fmt("hill")],
    ["Q-learning (ours)", *fmt("q")],
], widths=[2.0, 1.6, 1.5, 1.6])
doc.add_picture(os.path.join(OUT, "policy_paths.png"), width=Inches(4.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Figure 2. Hidden yield landscape with greedy paths of the trained agent from four starting recipes.",
     italic=True, size=9)
doc.add_heading("Discussion", level=2)
q, h, rd = E["q"], E["hill"], E["random"]
para(f"Q-learning converges to the high-yield region in {q['pct_reaching_global_opt']:.0f}% of evaluation "
     f"campaigns (converged yield {q['final_yield_mean']:.1f}%), compared with "
     f"{h['pct_reaching_global_opt']:.0f}% for the hill-climbing heuristic "
     f"({h['final_yield_mean']:.1f}%) and {rd['pct_reaching_global_opt']:.0f}% for random search "
     f"({rd['final_yield_mean']:.1f}%). The heuristic often gets trapped on the decoy local optimum at "
     "low temperature, or is misled by measurement noise, because it only ever looks one step ahead. "
     "The RL agent learns, from many past campaigns, which direction leads to the global peak — even "
     "crossing the low-yield valley that a one-step heuristic will never cross (Figure 2). It still settles on "
     "the decoy when it starts right next to it, because the state does not record what has already been "
     "explored. It also learns to stop at ~90 °C rather than pushing into the "
     "decomposition region, because the energy cost and falling yield are reflected in the return.")
doc.add_heading("Limitations", level=2)
bullet("Starting recipes immediately beside the decoy peak can still settle there: the state is only the "
       "current recipe, with no memory of the experiments already run.")
bullet("The agent learns one fixed landscape; a new reaction would require retraining (it memorises, it does not "
       "generalise). Function approximation (DQN) or meta-learning would be needed to transfer.")
bullet("Training used 3000 × 30 = 90,000 simulated experiments — far more than a real lab could afford. "
       "In practice the agent should be pre-trained on a simulator or combined with Bayesian optimisation.")
bullet("Coarse 11 × 11 grid and only two variables; real recipes have many continuous parameters "
       "(time, catalyst, solvent, pH) which makes tabular methods infeasible.")
bullet("Simulator is synthetic; noise and the sim-to-real gap are much larger in real chemistry.")
doc.add_heading("Ethics", level=2)
bullet("Safety: an agent maximising yield may propose hazardous conditions (high temperature/pressure, "
       "toxic reagents). Hard safety constraints and human approval are required — we model this only as a soft cost.")
bullet("Dual use: the same optimisation loop could be applied to harmful compounds; access control and "
       "screening of target molecules are needed.")
bullet("Accountability and reproducibility: autonomous decisions must be logged so that scientists can audit "
       "and reproduce results; energy use and waste should be part of the reward.")

# ---------------- Appendix ----------------
p = doc.add_paragraph(); p.add_run().add_break(WD_BREAK.PAGE)
doc.add_heading("Appendix: Source Code (lab_rl.py)", level=1)
code = open(os.path.join(HERE, "lab_rl.py")).read()
for line in code.splitlines():
    p = doc.add_paragraph()
    r = p.add_run(line if line else " "); r.font.name = "Consolas"; r.font.size = Pt(7.5)
    p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0

doc.save(os.path.join(HERE, "RL_Project7_Report_Moosa_Siddharth.docx"))

# =============================================================================
# SLIDES (5)
# =============================================================================
prs = Presentation()
prs.slide_width, prs.slide_height = PI(13.333), PI(7.5)
BLANK = prs.slide_layouts[6]
NAVY = PRGB(0x1F, 0x3A, 0x5F); ACC = PRGB(0x1F, 0x77, 0xB4); GREY = PRGB(0x55, 0x55, 0x55)


def slide(num, heading):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, PI(1.05)); bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tf = bar.text_frame; tf.margin_left = PI(0.5)
    r = tf.paragraphs[0].add_run(); r.text = f"{num}. {heading}"
    r.font.size = PPt(32); r.font.bold = True; r.font.color.rgb = PRGB(255, 255, 255)
    ft = s.shapes.add_textbox(PI(0.5), PI(7.0), PI(12.3), PI(0.4)).text_frame
    r = ft.paragraphs[0].add_run()
    r.text = f"{TITLE}  |  " + ", ".join(n for n, _ in STUDENTS) + f"  |  {num}/5"
    r.font.size = PPt(11); r.font.color.rgb = GREY
    return s


def bullets(s, items, x=0.5, y=1.35, w=12.3, h=5.5, size=20):
    tf = s.shapes.add_textbox(PI(x), PI(y), PI(w), PI(h)).text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sub = it.startswith("  ")
        r = p.add_run(); r.text = ("– " if sub else "• ") + it.strip()
        r.font.size = PPt(size - 3 if sub else size); r.font.color.rgb = GREY if sub else PRGB(0x22, 0x22, 0x22)
        p.level = 1 if sub else 0
        p.space_after = PPt(8)
    return tf


# Cover slide
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid()
bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()


def cover_line(text, y, size, bold=False, color=PRGB(255, 255, 255), h=0.8):
    tf = s.shapes.add_textbox(PI(0.5), PI(y), PI(12.3), PI(h)).text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = 2  # center
    r = p.add_run(); r.text = text; r.font.size = PPt(size); r.font.bold = bold; r.font.color.rgb = color


cover_line("Reinforcement Learning — Unit 5 Homework Project", 0.9, 20, color=PRGB(0xC8, 0xD6, 0xE5))
cover_line(TITLE, 1.8, 40, bold=True, h=1.0)
cover_line(SUB, 3.0, 20, color=PRGB(0xC8, 0xD6, 0xE5), h=1.0)
cover_line("Submitted by", 4.2, 20, bold=True)
for k, (n, roll) in enumerate(STUDENTS):
    cover_line(f"{n}  —  Roll No. {roll}", 4.8 + 0.55 * k, 22)

# 1 Motivation
s = slide(1, "Motivation — Self-Driving Labs")
bullets(s, [
    "Discovering new materials / reactions = searching a huge space of recipes",
    "  Each experiment costs hours, reagents and money; results are noisy",
    "Self-driving labs: robots run experiments, an AI decides what to try next",
    "  e.g. autonomous chemistry & materials platforms (A-Lab, mobile robotic chemist)",
    "Question: can an RL agent learn a strategy that finds the best reaction conditions "
    "within a fixed budget of experiments — better than random search or a simple heuristic?",
    "Our task: choose temperature (20–120 °C) and concentration (0.1–2.1 M) to maximise yield",
], w=12.3)

# 2 Method
s = slide(2, "Method — MDP + Q-learning")
bullets(s, [
    "Environment: simulated reaction, hidden yield surface",
    "  Global optimum ≈ 90 °C / 1.5 M, decoy local optimum ≈ 40 °C / 0.5 M, decomposition > 100 °C, noise σ = 3%",
    "State: current recipe on 11 × 11 grid (121 states)",
    "Actions (5): T ±10 °C, C ±0.2 M, repeat recipe",
    "Reward: measured yield/100 − energy cost for hot runs",
    "Episode: 30-experiment campaign from a random start",
    "Algorithm: tabular Q-learning, α = 0.5/√N (visit-decayed), γ = 0.97, ε-greedy 1.0→0.05",
    "Baselines: random policy; hill-climbing heuristic",
    "3000 training episodes × 5 seeds",
], w=7.0, size=18)
s.shapes.add_picture(os.path.join(OUT, "policy_paths.png"), PI(7.6), PI(1.4), width=PI(5.4))

# 3 Results
s = slide(3, "Results — Learning Curves vs. Baselines")
s.shapes.add_picture(os.path.join(OUT, "learning_curves.png"), PI(0.4), PI(1.25), width=PI(8.4))
bullets(s, [
    f"Q-learning: {q['final_yield_mean']:.1f}% converged yield, reaches global optimum in "
    f"{q['pct_reaching_global_opt']:.0f}% of campaigns",
    f"Hill-climbing: {h['final_yield_mean']:.1f}% ({h['pct_reaching_global_opt']:.0f}%) — trapped by local optimum / noise",
    f"Random: {rd['final_yield_mean']:.1f}% ({rd['pct_reaching_global_opt']:.0f}%)",
    "Agent overtakes the heuristic after ~700 episodes, as ε decays",
], x=8.9, y=1.4, w=4.2, size=16)
s.shapes.add_picture(os.path.join(OUT, "evaluation_bar.png"), PI(1.5), PI(4.5), height=PI(2.45))

# 4 Limitations
s = slide(4, "Limitations")
bullets(s, [
    "Learns one fixed reaction — no generalisation to a new chemistry without retraining",
    "Sample-hungry: 90,000 simulated experiments ≫ a real lab budget",
    "  Real use: pre-train in simulation, or combine with Bayesian optimisation / surrogate models",
    "Tabular method: only 2 variables on a coarse grid",
    "  Real recipes: many continuous parameters → need DQN / PPO with function approximation",
    "Synthetic simulator: sim-to-real gap, real noise and drift are larger",
    "Starts right next to the decoy peak can still settle there (state has no memory of what was explored)",
    "Safety handled only as a soft penalty, not a hard constraint",
])

# 5 Ethics
s = slide(5, "Ethics & Conclusion")
bullets(s, [
    "Safety: yield-maximising agents may propose hazardous conditions → hard constraints + human sign-off",
    "Dual use: same loop could optimise harmful substances → screening & access control",
    "Accountability: log every autonomous decision for audit and reproducibility",
    "Sustainability: include energy use, waste and solvent toxicity in the reward",
    "Conclusion: a simple Q-learning agent learned a campaign strategy that beats random search "
    "and a greedy heuristic, crossing the low-yield valley to the global optimum in most campaigns — a small-scale demonstration "
    "of how RL can drive autonomous experimentation",
])

prs.save(os.path.join(HERE, "RL_Project7_Presentation_Moosa_Siddharth.pptx"))
print("built report and slides")
