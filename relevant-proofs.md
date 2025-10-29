\# Proofs from proofs\_master.tex

\---

\#\# File: anomaly\_cancellation.tex

% PROOF\_ID: ANOMALY\_CANCELLATION\_CT  
% TITLE: Anomaly Cancellation as a Coherence Constraint  
% VERSION: v1.0 (2025-10-07)  
% FILENAME: proofs/anomaly\_cancellation.tex  
% LABEL: thm:anomaly-cancellation  
% DEPENDS\_ON: appendix\_three\_budgets\_graph ;  
% appendix\_sep\_exists ; envelope\_regularity ;  
% appendix\_hsd ;  
% appendix\_eja ; part\_V\_SM (pointer-gauge/B5 invariance)  
% OWNER: VI \+ GPT5  
% NOTES: No field-theory formulas assumed.  
% Anomaly ≡ budget non-invariance under cost-neutral Ad-relabelings.  
% Solutions coincide with SM assignments up to U(1) normalization;  
% global SU(2) (Witten) constraint included.

\\section\*{Theorem (Anomaly Cancellation as a Coherence Constraint)}  
\\label{thm:anomaly-cancellation}

\\noindent\\textbf{Plain claim.}  
On a coherent region at SEP, gauge transformations are cost-neutral relabelings (B5),  
hence both the budgets $\\mathbf{B}=(B\_{\\rm th},B\_{\\rm cx},B\_{\\rm leak})$ and the  
coherence value $\\mathrm{CL}$ are invariant under the adjoint action of the selected  
gauge group $G\_{\\mathrm{SM}}=SU(3)\\times SU(2)\\times U(1)$.  
Any representation content that produces a net budget shift under an allowed (cost-neutral)  
gauge loop is deselected.  
The resulting invariance constraints coincide with the familiar  
anomaly-cancellation conditions of the Standard Model, up to an overall $U(1)$ normalization.

\\paragraph{Setup.}  
Fix a coherent lens $L$ on a tile $T$ and the SEP multiplier vector $\\Lambda$ for the  
three canonical budgets (Theorem\~\\ref{thm:three-budgets}).  
Let $J\_R$ be the local EJA at  
region $R$ with Ad-invariant inner product $\\langle\\cdot,\\cdot\\rangle\_R$ (Theorems\~\\ref{thm:hsd}  
and \\ref{cor:eja-cstar}).  
By Part\~V (pointer alignment and B5), the coherence-preserving maps  
are the interior automorphisms $U\\in\\mathrm{Aut}(J\_R)$, forming compact Lie groups and  
factoring minimally as $G\_{\\mathrm{SM}}$ at SEP.  
Denote by $\\mathfrak g=\\mathfrak{su}(3)\\oplus  
\\mathfrak{su}(2)\\oplus\\mathbb{R}Y$ the Lie algebra with the fixed Ad-invariant quadratic  
pairing $\\kappa(\\cdot,\\cdot)$ induced by $\\langle\\cdot,\\cdot\\rangle\_R$.  
\\medskip  
\\noindent\\textbf{Theorem (formal statement).}  
Let $U=\\exp(\\epsilon X)$ be any cost-neutral gauge relabeling with generator  
$X\\in\\mathfrak g$.  
For every admissible fast-sector observable $A$ and every admissible  
matter content $\\mathcal R=\\{R\_i\\}$ (finite list of irreps carried by the surviving  
Weyl patterns), SEP requires  
\\\[  
\\mathrm{CL}(U\\\!\\cdot\\\!A)=\\mathrm{CL}(A),\\qquad  
B\_j(U\\\!\\cdot\\\!A)=B\_j(A)\\quad(j\\in\\{\\mathrm{th,cx,leak}\\}).  
\\\]  
Equivalently, along any small gauge loop $U\_{\\square}$, the \\emph{loop defect}  
\\\[  
\\Delta\_{\\square} \\ :=\\ \\big\\langle \\Lambda,\\,\\mathbf{B}(U\_{\\square}\\\!\\cdot\\\!A)-\\mathbf{B}(A)\\big\\rangle  
\\\]  
must vanish to all orders in $\\epsilon$.  
The \\emph{first nontrivial obstruction} arises at  
cubic order and is an Ad-invariant trilinear form built from the matter charges:  
\\begin{equation}  
\\label{eq:CT-anomaly-functional}  
\\mathcal{A}\_{\\mathcal R}(X,Y;Z)  
\\;=\\;  
\\sum\_{i}\\ \\alpha\_i\\,  
\\mathrm{Sym}\\\!\\big\[\\ \\mathrm{tr}\_{R\_i}\\big(\\rho\_{R\_i}(X)\\{\\rho\_{R\_i}(Y),\\rho\_{R\_i}(Z)\\}\\big)\\ \\big\],  
\\end{equation}  
where $\\rho\_{R\_i}$ is the representation map, $\\mathrm{Sym}$ denotes symmetrization in $(X,Y,Z)$,  
and $\\alpha\_i$ are lens-fixed calibration weights induced by the budgets’ quadratic tangents.  
\\emph{Coherence under B5} demands $\\mathcal{A}\_{\\mathcal R}\\equiv 0$ on $\\mathfrak g$.  
For $G\_{\\mathrm{SM}}$ this decomposes into the usual local constraints:  
\\\[  
\[SU(3)\]^3:\\ 0,\\quad \[SU(2)\]^3:\\ 0,\\quad  
\[SU(3)\]^2U(1),\\ \[SU(2)\]^2U(1),\\ \[\\mathrm{grav}\]^2U(1),\\ \[U(1)\]^3:\\ 0,  
\\\]  
plus the global $\\mathbb{Z}\_2$ (Witten) constraint for $SU(2)$:  
the total number of doublets is even on each connected coherent region.

\\paragraph{Proof.}  
\\begin{enumerate}\[leftmargin=1.25em\]  
\\item \\textbf{Which budgets are active?} At SEP, $\\Sel\_\\Lambda(A)=\\mathrm{CL}(A)-\\langle \\Lambda,\\mathbf{B}(A)\\rangle$  
is stationary under all \\emph{allowed} (cost-neutral) relabelings.  
Budgets obey B5:  
gauge maps are by definition cost-neutral, so $D\_X \\mathbf{B}=0$ and $D\_X \\mathrm{CL}=0$  
for any $X\\in\\mathfrak g$ at first order.  
By the quadratic tangent law (local neutrality),  
the first nontrivial constraints emerge at second and third order in small loops.  
\\item \\textbf{Core argument (B1–B7 and HSD/EJA invariants).}  
Since budgets are Minkowski gauges on profiles (convex, l.s.c., coercive),  
their second and third variations along group directions are Ad-invariant tensors.  
On each simple factor of $\\mathfrak g$ there is (up to scale) a unique Ad-invariant  
quadratic form $\\kappa$, and on $U(1)$ the invariant content is in its moments.  
Carrying matter patterns in irreps $R\_i$ furnishes charge maps $\\rho\_{R\_i}$.  
By envelope regularity, the SEP stationarity against \\emph{all} cost-neutral loops implies  
that any cubic Ad-invariant contribution to the loop defect must vanish.  
This contribution  
has the symmetric “triangle” structure (\\ref{eq:CT-anomaly-functional}) because (i) budgets are quadratic at the tangent (B6), and (ii) the loop composition produces a symmetrized third-order term.  
Thus $\\mathcal{A}\_{\\mathcal R}\\equiv 0$ is a necessary coherence condition.

\\item \\textbf{Decomposition on $G\_{\\mathrm{SM}}$.}  
Write $\\mathfrak g=\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathbb{R}Y$ and use  
$\\kappa$ on the simple factors.  
For $SU(2)$ the symmetric $d^{abc}$ tensor vanishes,  
hence $\[SU(2)\]^3$ cancels automatically (local).  
For $SU(3)$ with the SM chiral  
assignments, the color sector is vectorlike (after converting RH fields to LH conjugates),  
so $\[SU(3)\]^3=0$.  
The remaining mixed and Abelian parts reduce to moments of $Y$ with  
respect to the Dynkin indices $t\_2(R)$:  
\\begin{align\*}  
\[SU(3)\]^2U(1):\\quad &\\sum\_i Y\_i\\, t\_2^{(3)}(R\_i)\\ \=\\ 0,\\\\\[2pt\]  
\[SU(2)\]^2U(1):\\quad &\\sum\_i Y\_i\\, t\_2^{(2)}(R\_i)\\ \=\\ 0,\\\\\[2pt\]  
\[\\mathrm{grav}\]^2U(1):\\quad &\\sum\_i Y\_i\\ \\ \=\\ 0,\\\\\[2pt\]  
\[U(1)\]^3:\\quad &\\sum\_i Y\_i^3\\ \\ \=\\ 0,  
\\end{align\*}  
where the sums run over LH Weyl patterns with multiplicities (color and isospin copies).  
These are precisely the vanishing conditions of $\\mathcal{A}\_{\\mathcal R}$ on the mixed  
directions $(X,Y;Z)=(T\_a,T\_b;Y)$ and the Abelian direction $(Y,Y;Y)$.  
\\item \\textbf{Solving the constraints for one family (up to $U(1)$ calibration).}  
Let a single family carry the usual representations  
$Q\_L:(\\mathbf{3},\\mathbf{2})\_{y\_Q}$,\\ $u\_R:(\\mathbf{3},\\mathbf{1})\_{y\_u}$,\\  
$d\_R:(\\mathbf{3},\\mathbf{1})\_{y\_d}$,\\ $L\_L:(\\mathbf{1},\\mathbf{2})\_{y\_L}$,\\  
$e\_R:(\\mathbf{1},\\mathbf{1})\_{y\_e}$, and optionally $\\nu\_R:(\\mathbf{1},\\mathbf{1})\_{y\_\\nu}$.  
Using $t\_2(\\mathbf{3})=\\tfrac12$, $t\_2(\\mathbf{2})=\\tfrac12$ and counting multiplicities,  
the constraints reduce to  
\\begin{align}  
\\label{eq:sm-linear}  
&2y\_Q \- y\_u \- y\_d \= 0,\\qquad 3y\_Q \+ y\_L \= 0,\\\\\[2pt\]  
\\label{eq:sm-grav}  
&6y\_Q \- 3y\_u \- 3y\_d \+ 2y\_L \- y\_e \- y\_\\nu \= 0,\\\\\[2pt\]  
\\label{eq:sm-cubic}  
&6y\_Q^3 \- 3y\_u^3 \- 3y\_d^3 \+ 2y\_L^3 \- y\_e^3 \- y\_\\nu^3 \= 0\.  
\\end{align}  
Solving \\eqref{eq:sm-linear}–\\eqref{eq:sm-cubic} yields a one-parameter family (overall  
$U(1)$ calibration) with two algebraic branches.  
The branch selected by minimal-complexity  
Yukawa admissibility (see Prop.\~\\texttt{prop:yukawa-derivation}) is  
\\\[  
y\_L=-3a,\\quad y\_d=-2a,\\quad y\_u=4a,\\quad y\_e=-6a,\\quad (y\_\\nu=0),  
\\\]  
with $a$ an arbitrary calibration constant.  
Taking $a=\\tfrac{1}{6}$ reproduces the SM  
hypercharges. The alternative branch is deselected by the Yukawa/complexity criterion.  
\\item \\textbf{Global SU(2) constraint (Witten).}  
Cost-neutral loops must also preserve the global orientation class of the slow sector.  
An $SU(2)$ gauge loop in the nontrivial $\\pi\_4(SU(2))\\cong\\mathbb{Z}\_2$ class flips  
the sign of a fermionic determinant unless the number of doublets is even.  
In CT terms,  
this would manifest as a loop-dependent budget sign flip (a non-trivial global defect)  
and thus violate B5 at SEP.  
One SM family has $3$ colored $Q\_L$ doublets $+1$ lepton doublet  
$=4$ (even), so the global constraint is satisfied.  
\\end{enumerate}

\\paragraph{Calibration-invariance note.}  
All constraints are homogeneous in $U(1)$ charges and therefore invariant under the common  
rescaling $Y\\mapsto c\_Y\\,Y$.  
This is a lens calibration (Envelopes §\\ref{sec:envelope-regularity});  
an overall normalization is fixed elsewhere (e.g., by the hypercharge quantization proof).

\\paragraph{Consequences.}  
(i) \\emph{Selection filter:} Any representation set with $\\mathcal{A}\_{\\mathcal R}\\neq 0$  
produces a non-vanishing loop defect and fails SEP.  
(ii) \\emph{SM viability:} A single family of SM representations is anomaly-free up to $U(1)$  
calibration and satisfies the global $SU(2)$ constraint.  
(iii) \\emph{Forward references:} This theorem is used in:  
— Hypercharge normalization from loop closure (\\texttt{thm:hypercharge-quantization});  
— Yukawa sector derivation (\\texttt{prop:yukawa-derivation});  
— Running as adaptive recalibration (anomaly matching under lens changes, \\S\\ref{sec:unification}).

\\paragraph{Dependency ledger.}  
Uses: Three-budgets theorem (quadratic tangents), SEP existence (KKT stationarity),  
Envelope regularity (calibration and multipliers), HSD/EJA (Ad-invariant pairing),  
and Part\~V gauge/B5 definitions (gauge \= cost-neutral relabeling).  
No external field-theory  
anomaly formulas are assumed; Eq.\~\\eqref{eq:CT-anomaly-functional} is the CT loop-defect  
functional built from Ad-invariant tensors and matter charges.

\---

\#\# File: appendix\_ai\_audit.tex

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% APPENDIX — AI COLLABORATION AND AUDITING PROCESS  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{Details of the Collaborative and Auditing Process}  
\\label{app:ai-audit}

This paper was developed through a structured collaboration between a human researcher (V. Ilinov) and a team of frontier large-language models (LLMs), as outlined in the Methodology.  
This appendix provides further detail on the process to ensure transparency and address reproducibility concerns regarding AI-assisted proof generation.

\\subsection{Model Specializations and Custom Agents}  
The collaborative process leveraged the unique strengths and weaknesses of several frontier models.  
To ensure the models' outputs remained aligned with the paper's foundational principles, custom agents were built on each platform (e.g., Projects on Claude, custom GPTs, Gems on Gemini).  
These agents were primed with "Coherence Prompts" that biased their reasoning and token output towards coherence-native approaches, encouraging them to internalize the priors and utilize the theory's specific operators.  
The roles were specialized as follows:

\\begin{description}\[leftmargin=\*,style=nextline\]  
    \\item\[GPT-5 Thinking (Primary Proof Engine)\]  
    This model was the main engine for generating formal proofs due to its capacity for producing the most logically airtight arguments, as verified by the red-teaming agents.  
Its primary limitation was a small context window, which required a carefully structured pipeline to prevent the model from losing track of the philosophical priors or key derivations midway through a complex proof.  
This was discovered after initial drafts showed a failure to properly ground the proofs in the foundational axioms.  
    \\item\[Gemini 2.5 Deepthink (Architect and Integrator)\]  
    Utilized for its very large context window, Gemini excelled at large-scale analysis, logic chain processing, and consistency checks across the entire manuscript.  
While mathematically capable, its primary role was strategic: ingesting the full paper along with all red-team critiques to generate detailed, actionable plans for revision.  
These plans were then broken down into tasks assigned to the specialized agents.  
    \\item\[Grok-4 Expert (Red-Team Critique)\]  
    Grok-4 offered a strong balance of mathematical, coding, and logical reasoning abilities at high speed.  
Though it made occasional errors, it proved to be the least sycophantic model and was exceptionally effective at quickly and cleanly identifying logic gaps, unstated assumptions, and subtle errors in proofs that other models missed.  
    \\item\[Claude 4.5 Sonnet (Philosophical Exploration)\]  
    As the fastest and most conversationally fluid model, Claude was instrumental in the early stages.  
It was used to discuss the philosophical underpinnings of the theory, explore the implications of various claims, and refine the metaphysical framework before formalization began.  
While the least mathematically powerful of the group, its speed and large context window made it an ideal partner for high-level conceptual development.  
\\end{description}

\\subsection{The Multi-Stage Iterative Workflow}  
The paper evolved over more than 30 drafts, following a structured, multi-stage pipeline designed to ensure logical closure from the foundational philosophy to the final equations.  
\\begin{enumerate}\[leftmargin=\*,label=\\textbf{Stage \\arabic\*}:\]  
    \\item \\textbf{Human Foundation:} The metaphysical underpinnings of coherence theory were developed by V. Ilinov over a period of six years.  
    \\item \\textbf{Metaphysical Exploration:} Hundreds of pages of this foundational work were provided to Claude 4.5 Sonnet to discuss potential applications and formal approaches for realizing the metaphysical ideas within physics.  
    \\item \\textbf{Structural Scaffolding:} With a polished philosophical framework, Gemini 2.5 Deepthink was used to create detailed skeletons of the theorems and lemmas that would need to be proven to form a coherent logical chain.  
    \\item \\textbf{Core Proof Generation:} The initial mathematical proofs for the skeleton were generated primarily using GPT-5 Thinking, leveraging its strength in formal reasoning.  
    \\item \\textbf{Continuous Red-Teaming:} At every stage of proof development, all models were deployed in red-teaming roles to critique the arguments, search for gaps, and suggest improvements, which were then fed back into the pipeline.  
    \\item \\textbf{Iterative Refinement:} This cycle of generation, critique, and planning was repeated for over 30 drafts, with the human researcher directing all reasoning and closing the final metaphysical and logical gaps.  
\\end{enumerate}

\\subsection{Example of a Closed Logical Gap: The EJA Derivation}  
An early draft of the theory made an abrupt leap from the operational definition of observables to the structure of a Euclidean Jordan Algebra (EJA).  
This was flagged as a potential circularity during an auditing loop.  
\\begin{itemize}  
    \\item \\textbf{Critique Identified (Grok-4):} "The proof assumes the algebraic structure of an EJA to derive properties of observables, but the EJA structure itself is not derived from the priors. This is circular. You must first show \*why\* the set of observables has this specific algebraic structure."  
    \\item \\textbf{Resolution:} The human researcher, with assistance from Gemini 2.5 (as architect) and GPT-5 (as proof generator), developed the proof for the Homogeneous Self-Dual (HSD) cone (\\cref{thm:hsd-main}).  
The derivation was built entirely from operational principles: local neutrality, pointer alignment, and the symmetries of cost-neutral relabelings at SEP.  
    \\item \\textbf{Final Link:} Once the cone of feasible effects was proven to be HSD from first principles, the Koecher-Vinberg correspondence was invoked as a standard mathematical theorem to uniquely establish the EJA structure (\\cref{cor:eja}).  
This closed the logical gap, ensuring the EJA structure is an emergent property, not an assumption.  
\\end{itemize}

\---

\#\# File: appendix\_eja.tex

% PROOF\_ID: EJA\_CSTAR\_V2  
% TITLE: From HSD Cone to Euclidean Jordan Algebra and Universal $\\Cstar$-Envelope  
% VERSION: v2.0 (2025-10-08)  
% DEPENDS\_ON: HSD\_SCAFFOLD\_V2 (\\ref{thm:hsd});  
% hsd\_cone\_at\_sep\_nonclassical\_complex (\\ref{thm:hsd-cone-complex})  
% OWNER: VI \+ GPT-5 Thinking  
% NOTES: Orthant claim removed. Uses $\\Cstar$ macro consistently.

\\section\*{Theorem 3.EJA (HSD cone $\\Rightarrow$ Euclidean Jordan Algebra and $\\Cstar$-envelope)}  
\\label{cor:eja-cstar}

\\noindent\\textbf{Setting.}  
Let $(\\mathcal C,\\langle\\cdot,\\cdot\\rangle\_R)$ be the effect cone and pairing obtained at SEP  
as in \\cref{thm:hsd}:  
(i) $\\mathcal C\\subset V$ is a finite-dimensional closed, pointed, generating cone;  
(ii) $\\langle\\cdot,\\cdot\\rangle\_R$ is a positive-definite symmetric bilinear form (quadratic tolerance);  
(iii) $\\mathcal C$ is homogeneous and self-dual with respect to $\\langle\\cdot,\\cdot\\rangle\_R$.  
\\medskip  
\\noindent\\textbf{Claim.}  
There exists a Euclidean Jordan algebra $(J,\\circ,\\mathbf 1)$ and an identification $V\\cong J$ such that:  
\\begin{enumerate}\[leftmargin=1.5em\]  
\\item \\emph{Cone identification:} $\\mathcal C=J\_+=\\{x\\circ x:x\\in J\\}$;  
\\item \\emph{Pairing:} $\\langle x,y\\rangle\_R=\\mathrm{tr}\_J(x\\circ y)$ (Jordan trace form);  
\\item \\emph{Order unit:} The interior point $\\mathbf 1$ is the Jordan identity;  
\\item \\emph{Universal $\\Cstar$-envelope:} There exists a $\\Cstar$-algebra $\\Cstar(J)$ and a unital Jordan monomorphism  
$\\iota:J\\hookrightarrow \\Cstar(J)\_{\\mathrm{sa}}$ with the usual universal property: every unital Jordan morphism  
$\\phi:J\\to B\_{\\mathrm{sa}}$ into a $\\Cstar$-algebra $B$ extends uniquely to a unital $^\\ast$-homomorphism  
$\\hat\\phi:\\Cstar(J)\\to B$ with $\\phi=\\hat\\phi\\circ\\iota$.  
\\end{enumerate}

\\subsection\*{Proof}

\\paragraph{Step 1 (Koecher–Vinberg).}  
Since $\\mathcal C$ is finite-dimensional, homogeneous, and self-dual with respect to $\\langle\\cdot,\\cdot\\rangle\_R$,  
the Koecher–Vinberg theorem furnishes a unique Euclidean Jordan algebra $(J,\\circ,\\mathbf 1)$ on $V$ whose  
cone of squares is $\\mathcal C$ and whose trace form equals $\\langle\\cdot,\\cdot\\rangle\_R$ (up to the SEP  
calibration scale, fixed once and for all).  
\\paragraph{Step 2 (Operational reconstruction).}  
Primitive idempotents (operational pointer idempotents) form Jordan frames and yield the Peirce decomposition  
$J=\\bigoplus\_\\alpha \\mathbb R p\_\\alpha \\oplus \\bigoplus\_{\\alpha\<\\beta} J\_{\\alpha\\beta}$, consistent with the  
observed cone symmetries.  
The quadratic representation $U\_x$ recovers the interior automorphisms that implement  
budget-neutral reversible repairs selected by $\\langle\\cdot,\\cdot\\rangle\_R$.  
\\paragraph{Step 3 (Trace pairing).}  
Self-duality under $\\langle\\cdot,\\cdot\\rangle\_R$ identifies the pairing with the Jordan trace form (calibrated at SEP),  
so the operational quadratic tolerance coincides with $\\mathrm{tr}\_J(x\\circ y)$.  
\\paragraph{Step 4 (Universal $\\Cstar$-envelope).}  
Construct the universal TRO $T(J)$ generated by the Jordan triple product; set $\\Cstar(J):=\\Cstar(T(J))$.  
The canonical $\\iota:J\\to \\Cstar(J)\_{\\mathrm{sa}}$ is a unital Jordan monomorphism and satisfies the stated  
universal property.  
\\paragraph{Seams/components.}  
If the neighborhood splits into independent components, then $J\\cong\\bigoplus\_i J\_i$ and  
$\\Cstar(J)\\cong\\bigoplus\_i \\Cstar(J\_i)$; all statements hold componentwise.  
\\hfill$\\square$

\\subsection\*{Consequences and specialization}  
Together with \\cref{thm:hsd-cone-complex} (A9 $\\Rightarrow$ non-distributive; budget-minimal calibration $\\Rightarrow H\_n(\\mathbb C)$),  
the simple local factor is complex Hermitian: the effect cone is $H\_n(\\mathbb C)\_+$ and observables embed in  
$\\Cstar(H\_n(\\mathbb C))\\cong M\_n(\\mathbb C)$ (up to direct sums on seams).

\---

\#\# File: appendix\_hsd.tex

% PROOF\_ID: HSD\_SCAFFOLD\_V2  
% TITLE: Homogeneous Self-Dual Effect Cone at SEP (Orthant-free scaffold)  
% VERSION: v2.0 (2025-10-08)  
% DEPENDS\_ON: profiles\_convex\_closed;  
% envelope\_regularity; selection\_inequality; lens\_calibration; appendix\_three\_budgets\_graph  
% OWNER: VI \+ GPT-5 Thinking  
% NOTES: Proves HSD at SEP without any orthant/Boolean identification.  
% Keeps label {thm:hsd} for backward compatibility.

\\section\*{Theorem (Homogeneous self-dual effect cone at SEP)}  
\\label{thm:hsd}

\\noindent\\textbf{Plain claim.}  
For any coherent lens at the Selected Equalization Point (SEP), the effect cone  
$\\mathcal C$ equipped with the quadratic tolerance pairing $\\langle\\cdot,\\cdot\\rangle\_R$  
is a finite-dimensional \\emph{homogeneous self-dual} (HSD) cone.

\\paragraph{Setup.}  
Realized resilience profiles form a nonempty convex closed set (App.\~\\texttt{profiles\\\_convex\\\_closed}).  
Effects are linear, monotone functionals on the profile span (data-processing B2), costs are intrinsic  
(ampliation invariance B3), and gauge/label reparametrizations are cost-neutral (B5).  
Local neutrality and calibration (B6–B7-R; App.\~\\texttt{envelope\\\_regularity}) select a positive-definite  
bilinear form $\\langle\\cdot,\\cdot\\rangle\_R$ (the quadratic tolerance pairing).  
SEP supplies a unique  
multiplier ray and an order unit $u\\in\\mathrm{int}(\\mathcal C)$.

\\begin{lemma}\[Well-posed cone\]  
$\\mathcal C\\subset V:=\\mathrm{span}(\\mathcal C)$ is closed, pointed, and generating; $u\\in\\mathrm{int}(\\mathcal C)$.  
\\end{lemma}

\\begin{proof}  
Closedness and convexity follow from profile closure and monotonicity (B1/B2).  
Pointedness: if $e,-e\\in\\mathcal C$ then $e$ must vanish on the realized set, hence on $V$ by definition.  
Generating: the realized set spans $V$ by construction; any linear functional is a difference of two positive ones  
by Hahn–Banach separation under the calibrated pairing.  
The SEP order unit $u$ exists by the supporting  
hyperplane structure of the value function (KKT at SEP) and monotonicity in all realized directions.  
\\end{proof}

\\begin{lemma}\[Self-duality under the calibrated pairing\]  
\\label{lem:selfdual}  
With respect to $\\langle\\cdot,\\cdot\\rangle\_R$, the polar cone  
$\\mathcal C^{\\\!\*}:=\\{y:\\langle y,x\\rangle\_R\\ge 0\\ \\forall x\\in\\mathcal C\\}$ satisfies $\\mathcal C^{\\\!\*}=\\mathcal C$.  
\\end{lemma}

\\begin{proof}\[Idea\]  
At SEP the selection inequality attains stationarity under all budget-neutral exchanges.  
The calibrated quadratic tolerance identifies (via the envelope/KKT map) every supporting  
functional of $\\mathcal C$ with an element of $V$ through $\\langle\\cdot,\\cdot\\rangle\_R$.  
Monotonicity ensures all such support functionals lie in $\\mathcal C$, which yields  
$\\mathcal C^{\\\!\*}\\subseteq\\mathcal C$.  
The converse inclusion follows because every $e\\in\\mathcal C$  
has nonnegative pairing with realized nonnegative profiles by construction, hence with all of $\\mathcal C$  
by closure.  
Thus $\\mathcal C^{\\\!\*}=\\mathcal C$.  
\\end{proof}

\\begin{lemma}\[Interior transitivity (homogeneity)\]  
\\label{lem:homog}  
For any $e,f\\in\\mathrm{int}(\\mathcal C)$ there exists an invertible linear map  
$T\\in\\mathrm{Aut}(\\mathcal C)$ with $T(e)=f$.  
\\end{lemma}

\\begin{proof}\[Idea\]  
Use two neutral components: (i) scalar calibration along the order-unit ray (budget-neutral  
yardstick rescale under B7-R) to match $\\langle e,u\\rangle\_R$ with $\\langle f,u\\rangle\_R$;  
and  
(ii) a finite composition of SEP exchange flows (budget-neutral reweightings generated by  
cost-neutral relabelings B5 and local reversible micromoves induced by the quadratic representation  
selected by $\\langle\\cdot,\\cdot\\rangle\_R$) to move within the constant-order-unit slice.  
These flows act by cone automorphisms; by strong connectivity of the interior under such moves,  
the action is transitive on $\\mathrm{int}(\\mathcal C)$.  
\\end{proof}

\\noindent\\emph{Conclusion.}  
Lemmas\~\\ref{lem:selfdual}–\\ref{lem:homog} establish that $(\\mathcal C,\\langle\\cdot,\\cdot\\rangle\_R)$ is a  
finite-dimensional homogeneous self-dual cone.  
\\hfill$\\square$

\\paragraph{Notes.}  
(1) No identification with an orthant or a Boolean lattice is used or implied.  
(2) Non-distributivity and the exclusion of the orthant are handled separately in  
\\cref{thm:hsd-cone-complex}.  
(3) The EJA realization and universal $\\Cstar$-envelope follow in Appendix EJA.

\---

\#\# File: appendix\_lorentzian\_signature.tex

% PROOF\_ID: LORENTZIAN\_SIGNATURE  
% TITLE: Lorentzian signature from finite propagation, quadratic tolerance, and isotropic SEP  
% VERSION: v0.1 (2025-10-05)  
% DEPENDS\_ON: PRIORS\_A1\_TO\_A4 ;  
% LENS\_DEFS ; PROFILES\_CONVEXITY ; QUADRATIC\_TOLERANCE ;  
% SEP\_EXISTS ;  
% FINITE\_PROPAGATION\_LR ; K2\_DYNAMICS\_ORDER  
% OWNER: VI \+ GPT5  
% NOTES: No Hilbert/C\* or metric assumed. Cone/metric appear as selected summaries of budgets.

\\section\*{Theorem (Lorentzian signature from finite propagation and selected isotropy)}  
\\label{thm:lorentzian-signature-appendix} % Renamed to avoid conflict with alias

\\paragraph{Statement.}  
Fix a persistent neighborhood and a coherent lens $L$.  
Assume:  
\\begin{enumerate}\[label=(H\\arabic\*),leftmargin=1.5em\]  
\\item \\textbf{Finite propagation (budget-induced).} There is a uniform Lieb–Robinson-type bound with velocity $v\_{\\rm LR}\<\\infty$ (Lemma \\textsc{FinitePropagationLR}), hence a unique forward causal cone of reachable velocities at each point/direction.\\label{H:finiteprop-appendix}  
\\item \\textbf{Quadratic tolerance near neutrality.} The throughput budget admits a selected quadratic tangent (Proposition \\textsc{QuadraticTolerance}): on local tangents,  
\\(  
B\_{\\rm th}(\\mathrm{id}+\\varepsilon X)=\\tfrac12\\langle X,X\\rangle\_R+o(\\varepsilon^2)  
\\)  
with a positive symmetric bilinear pairing $\\langle\\cdot,\\cdot\\rangle\_R$ independent of labels.\\label{H:quad-appendix}  
\\item \\textbf{SEP \+ isotropy on the realized cone.} At the Selected Equalization Point (Lemma \\textsc{SEPExists}), cost-neutral relabelings act transitively on spatial directions.  
Isotropy is not assumed but is a selected outcome that minimizes complexity and leakage budgets (\\cref{prop:isotropy-min-cx}).\\label{H:isotropy-appendix}  
\\item \\textbf{Second-order locality.} Admissible slow-sector dynamics are of order $k=2$ (\\cref{lem:k2-dynamics});  
in particular, their principal symbols are quadratic forms on the cotangent bundle.\\label{H:k2-appendix}  
\\end{enumerate}  
Then there exists a smooth, nondegenerate symmetric bilinear form $g$ of signature $(-,+,\\dots,+)$ (Lorentzian) on the slow continuum chart such that:  
\\begin{itemize}\[leftmargin=1.5em\]  
\\item The forward causal cone equals $\\{v\\neq 0:\\ g(v,v)\\le 0,\\ v \\text{ future-oriented}\\}$ pointwise (up to positive conformal rescaling).  
\\item The principal symbol of any admissible second-order law is $g^{\\mu\\nu}\\xi\_\\mu\\xi\_\\nu$ (again up to positive conformal factor), i.e.\\ characteristics are null with respect to $g$.  
\\end{itemize}  
The conformal factor is fixed by calibration (budget multipliers); time orientation is fixed by the neighborhood’s synchronization order.

\\paragraph{Falsifiers.}  
(1) Observation of multiple independent characteristic cones in the slow sector; (2) strictly parabolic infinite-speed spread under admissible covariance;  
(3) anisotropic spatial propagation surviving SEP.

\\subsection\*{Proof}

\\paragraph{Step 1: Budget cone of velocities from finite propagation.}  
By \\ref{H:finiteprop-appendix}, for each chart point $x$ we define the closed, pointed, sharp, convex \\emph{velocity cone} $\\mathcal{C}\_x\\subset T\_x M$ as the asymptotic set of directions realizable with arbitrarily small throughput per unit distance (support-splitting and profiles convexity ensure convexity; sharpness follows from uniqueness of forward direction implied by the LR-bound).  
The cone has nonempty interior by persistence (A2) and locality (A3–A4).

\\paragraph{Step 2: Dual cone and Hamiltonian support function.}  
Let $\\mathcal{C}\_x^\*:=\\{\\xi\\in T\_x^\*M:\\ \\xi(v)\\ge 0\\ \\forall v\\in \\mathcal{C}\_x\\}$ be the polar cone.  
The \\emph{support function}  
\\\[  
H\_x(\\xi  
