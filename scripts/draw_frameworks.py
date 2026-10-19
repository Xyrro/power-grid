"""Framework figures of docs/recursive/FRAMEWORKS.md, drawn as SVG (standard library only).

    python3 scripts/draw_frameworks.py          # writes docs/recursive/figures/*.svg

One colour per role, the same in every figure: inputs (grey), learned networks (blue), rules, decoders and repairs
(white), the LP / MILP solver (orange), results (dark). Boxes are placed on explicit coordinates; every label line is
checked against its box width, so a figure that would clip text fails to build.
"""
import os
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "recursive", "figures")
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
FS, LH = 14, 18                      # node text size and line height
CW, CWB = 7.3, 8.0                   # estimated advance per character, regular and bold, at 14 px
STYLE = {
    "input": dict(fill="#EEF1F5", stroke="#5B6B7F", text="#1F2A37"),
    "learned": dict(fill="#E3EDF9", stroke="#2F6DB5", text="#14325C"),
    "algo": dict(fill="#FFFFFF", stroke="#7A8594", text="#1F2A37"),
    "solver": dict(fill="#FCEBDD", stroke="#C8641E", text="#5A2A08"),
    "result": dict(fill="#1F2A37", stroke="#1F2A37", text="#FFFFFF"),
    "dropped": dict(fill="#F4F5F7", stroke="#A0A7B0", text="#6B7480", dash="5 4"),
    "note": dict(fill="#FBFBF8", stroke="#C9CED6", text="#3A4554"),
}
LEGEND = [("input", "input / signal"), ("learned", "learned network"), ("algo", "rule, decoder or repair"),
          ("solver", "LP / MILP solver"), ("result", "result")]
TAGS = {"added": "#2F6DB5", "kept": "#5B6B7F", "changed": "#3C7D6E", "dropped": "#9AA3AE"}
EDGE = {"main": "#3A4554", "solver": "#C8641E", "learned": "#2F6DB5", "dropped": "#9AA3AE"}


class Fig:
    def __init__(self, name, w, h, title):
        self.name, self.w, self.h, self.title = name, w, h, title
        self.boxes, self.body, self.kinds = {}, [], set()

    # ------------------------------------------------------------------ nodes
    def box(self, key, x, y, w, h, kind, lines, bold=0, shape="rect", tag=None):
        """lines: text lines; the first `bold` lines are bold"""
        st = STYLE[kind]
        self.boxes[key] = (x, y, w, h, shape)
        self.kinds.add(kind)
        dash = f' stroke-dasharray="{st["dash"]}"' if "dash" in st else ""
        if shape == "diamond":
            pts = f"{x + w / 2},{y} {x + w},{y + h / 2} {x + w / 2},{y + h} {x},{y + h / 2}"
            self.body.append(f'<polygon points="{pts}" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="1.5"{dash}/>')
            avail = w * 0.62
        else:
            self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{st["fill"]}" '
                             f'stroke="{st["stroke"]}" stroke-width="1.5"{dash}/>')
            avail = w - 18
        for i, ln in enumerate(lines):
            est = len(ln) * (CWB if i < bold else CW)
            if est > avail:
                raise SystemExit(f"{self.name}/{key}: line {ln!r} ~{est:.0f}px > {avail:.0f}px")
        if len(lines) * LH > h - 8:
            raise SystemExit(f"{self.name}/{key}: {len(lines)} lines do not fit in {h}px")
        y0 = y + h / 2 - (len(lines) - 1) * LH / 2 + 5
        for i, ln in enumerate(lines):
            wt = ' font-weight="700"' if i < bold else ""
            self.body.append(f'<text x="{x + w / 2}" y="{y0 + i * LH}" text-anchor="middle" font-size="{FS}" '
                             f'fill="{st["text"]}"{wt}>{escape(ln)}</text>')
        if tag:
            tw = len(tag) * 6.6 + 16
            tx, ty = x + w - tw - 8, y - 9
            self.body.append(f'<rect x="{tx}" y="{ty}" width="{tw}" height="18" rx="9" fill="{TAGS[tag]}"/>')
            self.body.append(f'<text x="{tx + tw / 2}" y="{ty + 13}" text-anchor="middle" font-size="11" '
                             f'font-weight="700" fill="#FFFFFF" letter-spacing="0.4">{tag.upper()}</text>')

    def frame(self, x, y, w, h, lines):
        self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#F7F9FC" stroke="#9FB3CC" '
                         f'stroke-width="1.2" stroke-dasharray="6 4"/>')
        for i, ln in enumerate(lines):
            wt = ' font-weight="700"' if i == 0 else ""
            self.body.append(f'<text x="{x + 14}" y="{y + 22 + i * 17}" font-size="13" fill="#2B4A72"{wt}>'
                             f'{escape(ln)}</text>')

    def text(self, x, y, lines, anchor="start", color="#3A4554", size=12.5, italic=True):
        st = ' font-style="italic"' if italic else ""
        for i, ln in enumerate(lines):
            self.body.append(f'<text x="{x}" y="{y + i * 16}" text-anchor="{anchor}" font-size="{size}" '
                             f'fill="{color}"{st}>{escape(ln)}</text>')

    # ------------------------------------------------------------------ anchors
    def L(self, k, t=0.5):
        x, y, w, h, _ = self.boxes[k]
        return (x, y + h * t)

    def R(self, k, t=0.5):
        x, y, w, h, _ = self.boxes[k]
        return (x + w, y + h * t)

    def T(self, k, t=0.5):
        x, y, w, h, _ = self.boxes[k]
        return (x + w * t, y)

    def B(self, k, t=0.5):
        x, y, w, h, _ = self.boxes[k]
        return (x + w * t, y + h)

    # ------------------------------------------------------------------ edges
    def edge(self, *pts, kind="main", dashed=False, head=True):
        c = EDGE[kind]
        d = " ".join(f"{'M' if i == 0 else 'L'}{p[0]:.1f},{p[1]:.1f}" for i, p in enumerate(pts))
        da = ' stroke-dasharray="6 4"' if dashed else ""
        mk = f' marker-end="url(#h-{kind})"' if head else ""
        self.body.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.6" stroke-linejoin="round"{da}{mk}/>')

    def label(self, x, y, text, anchor="start", color="#3A4554"):
        self.body.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="12.5" fill="{color}" '
                         f'font-style="italic">{escape(text)}</text>')

    # ------------------------------------------------------------------ output
    def legend(self, y):
        x = 20
        for k, txt in LEGEND:
            if k not in self.kinds:
                continue
            st = STYLE[k]
            self.body.append(f'<rect x="{x}" y="{y}" width="22" height="14" rx="3" fill="{st["fill"]}" '
                             f'stroke="{st["stroke"]}" stroke-width="1.5"/>')
            self.body.append(f'<text x="{x + 30}" y="{y + 12}" font-size="12.5" fill="#3A4554">{escape(txt)}</text>')
            x += 30 + len(txt) * 6.6 + 26

    def save(self):
        defs = "".join(f'<marker id="h-{k}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" '
                       f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                       for k, c in EDGE.items())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}" role="img" aria-label="{escape(self.title)}">'
               f'<title>{escape(self.title)}</title><defs>{defs}</defs>'
               f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>' + "".join(self.body) + "</svg>\n")
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"{self.name}.svg"), "w") as f:
            f.write(svg)
        print("wrote", os.path.join(OUT, f"{self.name}.svg"))


# ====================================================================== 1. the original two-stage framework
def two_stage():
    f = Fig("1_two_stage", 920, 760, "The original two-stage framework and what this study kept, added and dropped")
    f.box("I1", 20, 30, 270, 64, "input", ["Demand, renewables, reserve,", "previous on/off status"], tag="kept")
    f.box("I2", 325, 30, 270, 64, "input", ["One LP relaxation: fractional u,", "prices, line loadings"], tag="added")
    f.box("I3", 630, 30, 270, 64, "input", ["Labels: full-MILP labels", "replaced by polished labels"], tag="changed")
    f.box("M1", 325, 140, 270, 72, "learned", ["Model 1", "GNN; now a temporal-GNN ensemble"], bold=1, tag="kept")
    f.box("P", 325, 252, 270, 52, "algo", ["Commitment probabilities"])
    f.box("RP", 325, 344, 270, 64, "algo", ["Min up/down repair and", "block adequacy repair"], tag="added")
    f.box("U", 325, 448, 270, 52, "algo", ["On/off schedule u (units × hours)"])
    f.box("LP", 325, 540, 270, 64, "solver", ["Exact dispatch LP given u", "(the last step)"], tag="kept")
    f.box("OUT", 325, 644, 270, 52, "result", ["Cost, dispatch and angles"])
    f.box("M2", 630, 440, 270, 68, "dropped", ["Model 2: a network regressing", "dispatch and angles (MSE)"],
          tag="dropped")
    f.edge(f.B("I1"), (155, 116), (430, 116), f.T("M1", 105 / 270))
    f.edge(f.B("I2"), f.T("M1"))
    f.edge(f.B("I3"), (765, 116), (490, 116), f.T("M1", 165 / 270))
    for a, b in (("M1", "P"), ("P", "RP"), ("RP", "U"), ("U", "LP"), ("LP", "OUT")):
        f.edge(f.B(a), f.T(b))
    f.edge(f.R("U"), f.L("M2", 0.5), kind="dropped", dashed=True)
    f.edge(f.T("M2", 0.45), (751.5, 190), f.R("M1", 50 / 72), kind="dropped", dashed=True)
    f.text(762, 300, ["learned critic", "for Model 1:", "dropped"], color="#6B7480")
    f.edge(f.L("LP"), (200, 572), (200, 176), f.L("M1"), kind="solver", dashed=True)
    f.text(30, 330, ["REINFORCE with the", "exact LP as the critic:", "added"], color="#8A4310")
    f.legend(728)
    f.save()


# ====================================================================== 2. the MILP-based pipeline
def milp_pipeline():
    f = Fig("2_milp_pipeline", 920, 760, "The current best method: the MILP-based pipeline")
    f.box("X", 210, 20, 360, 56, "input", ["Instance: load, renewables, reserve", "requirement, initial status"])
    f.box("R", 620, 110, 280, 64, "solver", ["LP relaxation, solved once:", "input, guard and bound"])
    f.box("E", 210, 110, 360, 64, "learned", ["5-member temporal-GNN ensemble,", "trained on polished labels"])
    f.box("TH", 210, 214, 360, 64, "algo", ["Per-generator thresholds, tuned jointly", "on validation (Learning-to-Fix)"])
    f.box("G", 210, 318, 360, 64, "algo", ["Guards release risky fixings:", "adequacy, min up/down, LP relaxation"])
    f.box("M", 210, 422, 360, 64, "solver", ["Reduced MILP at a 0.5 % gap,", "about 89 % of decisions fixed"])
    f.box("D", 270, 524, 240, 96, "algo", ["Bound gap to the", "LP relaxation ≥ 4 %?"], shape="diamond")
    f.box("F", 20, 540, 200, 64, "solver", ["Also the full MILP at", "0.5 %; keep the cheaper"])
    f.box("OUT", 210, 662, 360, 56, "result", ["0.30 % mean gap, 97 % served,", "9.7× mean speed-up (test)"])
    f.edge(f.B("X"), f.T("E"))
    f.edge(f.R("X"), (760, 48), f.T("R"))
    f.edge(f.L("R"), f.R("E"), kind="solver")
    for a, b in (("E", "TH"), ("TH", "G"), ("G", "M"), ("M", "D")):
        f.edge(f.B(a), f.T(b))
    f.edge(f.B("R", 0.4), (732, 350), f.R("G"), kind="solver")
    f.edge(f.B("R", 0.75), (830, 572), f.R("D"), kind="solver")
    f.edge(f.L("D"), f.R("F"))
    f.label(262, 564, "yes", anchor="end")
    f.edge(f.B("D"), f.T("OUT"))
    f.label(400, 646, "no (99 % of test instances)")
    f.edge(f.B("F"), (120, 690), f.L("OUT"))
    f.text(586, 296, ["never fix a unit-hour", "off that the relaxation", "turns on"])
    f.legend(732)
    f.save()


# ====================================================================== 3. arm A: the TRM-style recursive model
def arm_a():
    f = Fig("3_arm_a_trm", 920, 1080, "Arm A: the TRM-style recursive model and its decoding pipeline")
    f.box("X", 120, 20, 680, 56, "input", ["Input tokens x: one per unit-hour (unit data, hour totals, bus PTDF code),",
                                           "plus the LP relaxation"])
    f.frame(20, 106, 880, 270, ["One supervision step, repeated up to N_sup times",
                                "(y, z) carried to the next step, detached; loss: BCE on every step’s answer + 0.5 × halting BCE"])
    f.box("Z", 60, 160, 300, 64, "learned", ["Latent update, n times:", "z ← f(x + y + z)"])
    f.box("Y", 60, 270, 300, 64, "learned", ["Answer update:", "y ← f(y + z)"])
    f.box("H1", 560, 160, 310, 64, "learned", ["Answer head: one logit", "per unit-hour"])
    f.box("H2", 560, 270, 310, 64, "learned", ["Halting head: stop when at most", "2 unit-hours are wrong"])
    f.edge(f.B("Z"), f.T("Y"))
    f.edge(f.R("Y", 0.3), (400, 289.2), (400, 192), f.R("Z"), kind="learned")
    f.text(410, 226, ["T cycles;", "gradient through", "the last only"])
    f.edge(f.R("Y", 0.75), (530, 318), (530, 192), f.L("H1"))
    f.edge((530, 302), f.L("H2"))
    f.text(60, 362, ["f: one shared 2-layer network (unit, time and graph mixing, SwiGLU)"], italic=True)
    f.box("P", 120, 410, 360, 52, "algo", ["Probabilities after k steps"])
    f.edge((300, 376), f.T("P"))
    f.box("W", 120, 496, 360, 52, "algo", ["Water-filling over identical units"])
    f.box("V", 120, 582, 360, 64, "algo", ["Per-unit Viterbi: min up/down times", "and initial status"])
    f.box("A", 120, 680, 360, 52, "algo", ["Block adequacy repair"])
    f.box("LP", 120, 766, 360, 52, "solver", ["Dispatch LP"])
    f.box("D", 180, 852, 240, 90, "algo", ["Sheds load or", "short of reserve?"], shape="diamond")
    f.box("LR", 560, 865, 310, 64, "algo", ["LP-guided repair: raise the failing", "hours’ reserve (≤ 2 rounds)"])
    f.box("S", 120, 976, 360, 64, "algo", ["Screen with 4 LP roundings,", "keep the cheapest"])
    for a, b in (("P", "W"), ("W", "V"), ("V", "A"), ("A", "LP"), ("LP", "D"), ("D", "S")):
        f.edge(f.B(a), f.T(b))
    f.edge(f.R("D"), f.L("LR"))
    f.label(436, 889, "yes")
    f.label(308, 962, "no")
    f.edge(f.T("LR"), (715, 792), f.R("LP"), kind="solver")
    f.label(724, 830, "reprice")
    f.box("PR", 560, 410, 310, 152, "note", ["Presets", "CPU: hidden 64, n = 4, T = 2,", "N_sup = 8 (135 k parameters)",
                                             "GPU: hidden 128, n = 6, T = 3,", "N_sup = 16 (0.63 M parameters)",
                                             "One step = T × (n + 1) passes of f"], bold=1)
    f.box("O", 560, 976, 310, 64, "result", ["Schedule and its cost", "(headline pipeline)"])
    f.edge(f.R("S"), f.L("O"))
    f.legend(1056)
    f.save()


# ====================================================================== 4. arm B: the LP in the loop
def arm_b():
    f = Fig("4_arm_b_rloop", 920, 900, "Arm B: recursion with the exact dispatch LP in the loop")
    f.box("X", 100, 20, 400, 56, "input", ["Tokens: unit-hour, bus and line inputs,", "plus the LP relaxation"])
    f.box("N", 100, 116, 400, 76, "learned", ["Recursive network", "latent updates, then an answer update;",
                                              "y and z carried between steps"], bold=1)
    f.box("H", 580, 20, 320, 56, "learned", ["Halting head: stop when the", "schedule is good enough"])
    f.box("HD", 100, 232, 400, 56, "learned", ["Count heads per group of identical", "units → scores"])
    f.box("DC", 100, 328, 400, 56, "algo", ["Viterbi decoding, block adequacy", "and min up/down repairs"])
    f.box("LP", 100, 424, 400, 56, "solver", ["Exact dispatch LP", "(warm-started within the instance)"])
    f.box("FB", 580, 276, 320, 132, "input", ["LP feedback", "hourly shed, spill, reserve shortfall;",
                                              "bus prices, line loadings, duals;", "unit dispatch, cost sensitivities;",
                                              "cost and certified gap"], bold=1)
    f.box("C", 100, 520, 400, 52, "algo", ["Priced candidate of step k"])
    f.box("BE", 100, 612, 400, 56, "algo", ["Keep the cheapest of K steps", "(more steps never hurt)"])
    f.box("SN", 100, 708, 400, 64, "solver", ["Safety net: not served, or certified gap", "≥ θ → reduced MILP at 0.5 %"])
    f.box("O", 100, 812, 400, 52, "result", ["Schedule and its cost"])
    f.edge(f.B("X"), f.T("N"))
    for a, b in (("N", "HD"), ("HD", "DC"), ("DC", "LP"), ("LP", "C"), ("C", "BE"), ("BE", "SN"), ("SN", "O")):
        f.edge(f.B(a), f.T(b))
    f.edge(f.R("N", 0.2), (540, 131.2), (540, 48), f.L("H"))
    f.edge(f.R("LP"), (740, 452), f.B("FB"), kind="solver")
    f.edge(f.T("FB"), (740, f.R("N", 0.74)[1]), f.R("N", 0.74), kind="solver", dashed=True)
    f.label(750, 230, "next step")
    f.box("PR", 580, 520, 320, 170, "note", ["Presets", "CPU: 166 k parameters, 2 latent", "updates per step, 3 training steps",
                                             "GPU: hidden 160, 3 latent updates", "× 2 cycles, 6 training steps,",
                                             "then 1,500 REINFORCE updates", "Control: every LP signal zeroed"], bold=1)
    f.legend(878)
    f.save()


# ====================================================================== 4b. arm B: LP-checked local search
def arm_b_search():
    f = Fig("5_arm_b_search", 920, 820, "Arm B, CPU round: local search with LP-checked moves ranked by the network")
    f.box("S0", 100, 20, 400, 56, "algo", ["Start: LP rounding and repairs,", "best of 5 LPs"])
    f.box("PR", 100, 116, 400, 56, "solver", ["Price the current best: cost and", "exact cost sensitivities (LP duals)"])
    f.box("MV", 100, 212, 400, 64, "algo", ["Enumerate block moves: switch a block on", "or off, trim an hour, follow the network"])
    f.box("RK", 100, 316, 400, 64, "algo", ["Rank: LP first-order Δcost", "− λ × network log-odds (λ = 0.5)"])
    f.box("NET", 600, 316, 300, 64, "learned", ["Recursive network, re-run on", "the current best’s LP feedback"])
    f.box("AP", 100, 420, 400, 56, "algo", ["Apply the best untried move;", "min up/down and adequacy repair"])
    f.box("PC", 100, 516, 400, 52, "solver", ["Price it with the dispatch LP"])
    f.box("D", 180, 600, 240, 80, "algo", ["Cheaper than", "the current best?"], shape="diamond")
    f.box("O", 100, 712, 400, 56, "result", ["After 16 LPs: the best schedule", "(+ safety net: reduced MILP)"])
    for a, b in (("S0", "PR"), ("PR", "MV"), ("MV", "RK"), ("RK", "AP"), ("AP", "PC"), ("PC", "D")):
        f.edge(f.B(a), f.T(b))
    f.edge(f.L("NET"), f.R("RK", 0.3), kind="learned")
    f.edge(f.L("D"), (40, 640), (40, 144), f.L("PR"))
    f.label(48, 404, "yes: accept")
    f.edge(f.R("D"), (540, 640), (540, 360.8), f.R("RK", 0.7))
    f.label(548, 470, "no: next move")
    f.edge(f.B("D"), f.T("O"))
    f.label(310, 700, "16 LPs used: stop")
    f.legend(790)
    f.save()


# ====================================================================== 5. the controls
def controls():
    f = Fig("6_controls", 920, 470, "The controls: each changes exactly one thing")
    f.box("M", 270, 178, 380, 92, "learned", ["Main models", "A: one shared network applied recursively",
                                              "B: the same idea, plus exact LP feedback"], bold=1)
    f.box("C1", 20, 20, 300, 64, "learned", ["A · one-shot: the same network,", "applied once"])
    f.box("C2", 600, 20, 300, 64, "learned", ["A · deep: unshared layers at the", "same depth, no recursion"])
    f.box("C3", 20, 364, 300, 64, "learned", ["A · no LP-relaxation inputs", "(recursive and one-shot)"])
    f.box("C4", 600, 364, 300, 64, "learned", ["B · no feedback: every", "LP-derived signal zeroed"])
    f.edge(f.L("M", 0.3), (170, 205.6), f.B("C1"))
    f.edge(f.R("M", 0.3), (750, 205.6), f.B("C2"))
    f.edge(f.L("M", 0.7), (170, 242.4), f.T("C3"))
    f.edge(f.R("M", 0.7), (750, 242.4), f.T("C4"))
    f.label(180, 130, "remove the recursion")
    f.label(740, 130, "remove the weight sharing", anchor="end")
    f.label(180, 320, "remove the LP inputs")
    f.label(740, 320, "remove the feedback", anchor="end")
    f.legend(446)
    f.save()


# ====================================================================== 6. the evaluation protocol
def protocol():
    f = Fig("7_protocol", 920, 866, "The evaluation protocol")
    f.box("V", 160, 20, 600, 56, "input", ["360 validation instances: model, decoder, thresholds,",
                                           "step count K and safety-net threshold chosen here"])
    f.box("F", 160, 112, 600, 44, "algo", ["The selection file is written; the test set is still unread"])
    f.box("T", 160, 192, 600, 56, "input", ["120 test_fresh instances, read once: per instance, one process",
                                            "on one core, single-threaded HiGHS and torch"])
    f.frame(100, 280, 720, 360, ["Back to back on the same core: paired timing"])
    f.box("M1", 160, 322, 600, 52, "solver", ["Full MILP at 0.1 % (60 s): dual bound DB and time T_MILP"])
    f.box("M2", 160, 400, 600, 44, "solver", ["Full MILP at 0.5 %"])
    f.box("R", 160, 470, 600, 44, "solver", ["LP relaxation, re-solved and timed (charged to every method using it)"])
    f.box("BM", 160, 540, 600, 72, "algo", ["Learning-free baselines, then the method under test,",
                                            "timed with every overhead: features, network passes,",
                                            "decoding, repairs, LPs and any MILP fallback"])
    f.box("P", 160, 672, 600, 56, "solver", ["Price every schedule with the exact dispatch LP (shedding",
                                             "$10,000/MWh, reserve shortfall $1,000/MWh, inside C)"])
    for a, b in (("V", "F"), ("F", "T"), ("T", "M1"), ("M1", "M2"), ("M2", "R"), ("R", "BM"), ("BM", "P")):
        f.edge(f.B(a), f.T(b))
    xs = (20, 245, 470, 695)
    names = (["Served: no shedding,", "no reserve shortfall"], ["Gap (C − DB) / C:", "mean, median, max"],
             ["Speed-up T_MILP / T:", "mean and geometric"], ["Paired bootstrap", "95 % CIs"])
    for i, (x, nm) in enumerate(zip(xs, names)):
        f.box(f"Q{i}", x, 772, 205, 56, "result", nm)
        f.edge((460, 728), (460, 750), (x + 102.5, 750), f.T(f"Q{i}"), head=True)
    f.legend(842)
    f.save()


if __name__ == "__main__":
    two_stage()
    milp_pipeline()
    arm_a()
    arm_b()
    arm_b_search()
    controls()
    protocol()
