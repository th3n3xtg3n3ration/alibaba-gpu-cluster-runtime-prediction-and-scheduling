%%
%% This "thesis.cls" file is a class file.
%% It provides you with definitions necessary to typeset
%% a thesis in the format required by the Graduate School,
%% Yeditepe University.
%%
%% File prepared by Osman Kerem Perente
%% <kperente@cse.yeditepe.edu.tr>
%%
%% Last modified on 14/04/2025
%%
%% Based on the original thesis template
%% prepared by Feza Kerestecioglu <kerestec@khas.edu.tr>
%% http://webprs.khas.edu.tr/~kerestec/fbe_tez.zip
%%
%% DO NOT CHANGE THIS FILE
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2024/09/14 LEE Thesis Class, Yeditepe University]

\typeout{LEE Thesis Class, Version 2.1}
\typeout{Prepared by Osman Kerem Perente, modified by Nezih Topaloglu}

% declare thesis class options
\global\let\@apkg\@ne
\global\let\@dtype\@ne
\global\let\@ptype\empty
\DeclareOption{algpseudocode}{\global\let\@apkg\@ne}
\DeclareOption{algorithm2e}{\global\let\@apkg\tw@}
\DeclareOption{msc}{\global\let\@dtype\@ne}
\DeclareOption{phd}{\global\let\@dtype\@ne \global\let\@ptype\@ne}
\DeclareOption{proposal}{\global\let\@dtype\tw@ \global\let\@ptype\tw@}
\DeclareOption{followup}{\global\let\@dtype\tw@ \global\let\@ptype\thr@@}
\DeclareOption*{\OptionNotUsed}
\ProcessOptions\relax

% thesis class is based on report class with 12pt
\LoadClass[a4paper,12pt]{report}

% load geometry, parskip, and setspace packages
\RequirePackage[a4paper]{geometry}
\RequirePackage[skip=10pt]{parskip}
\RequirePackage[singlespacing]{setspace}

% load algorithm packages according to the class options
\ifx\@apkg\@ne
    \RequirePackage{algpseudocode}
    \newcommand{\SetAlgLineNo}[1]{\setcounter{ALG@line}{\numexpr#1-1}}
\fi
\ifx\@apkg\tw@
    \RequirePackage[algo2e]{algorithm2e}
    \SetKwInput{KwRequire}{Require}
    \SetNlSty{textbf}{}{:}
\fi

% load math related packages
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{calc}
\RequirePackage{isomath}
\RequirePackage{upgreek}

% load figure and table related packages
\RequirePackage{booktabs}
\RequirePackage{enumitem}
\RequirePackage{graphicx}
\RequirePackage{hhline}
\RequirePackage{makecell}
\RequirePackage{multirow}
\RequirePackage{newfloat}

% load caption related packages
\RequirePackage{caption}
\RequirePackage{subcaption}

% load header and landscape related packages
\RequirePackage{fancyhdr}
\RequirePackage{pdflscape}

% load footnote related packages
\RequirePackage{fnpct}
\RequirePackage[bottom]{footmisc}

% load font related packages
\RequirePackage[T1]{fontenc}
\RequirePackage{newtxmath}
\RequirePackage{newtxtext}

% load listings related packages
\RequirePackage{listings}
\RequirePackage{xcolor}

% set paper margins
\ifx\@dtype\@ne
    \geometry{%
        top             =  3.5cm,
        bottom          =    2cm,
        left            =  3.5cm,
        right           =    2cm,
        headheight      =  1.5cm,
        headsep         =    1cm,
        footskip        =    1cm,
        marginparsep    =  .25cm,
        marginparwidth  =  1.5cm
    }
\else
    \geometry{%
        top             =  2.5cm,
        bottom          =  2.5cm,
        left            =  2.5cm,
        right           =  2.5cm,
        headheight      =    1cm,
        headsep         =  .75cm,
        footskip        = 1.25cm,
        marginparsep    =  .25cm,
        marginparwidth  =    2cm
    }
\fi

% 1.5 line spacing is used in the document
\setstretch{1.5}

% set page numbering style up to the introduction chapter
\pagenumbering{roman}

% adjust page style
\fancypagestyle{mypagestyle}{
    \fancyhf{}
    \rhead{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

% set page style
\pagestyle{mypagestyle}

% disable end of line hyphenation
\hyphenpenalty=\@M
\exhyphenpenalty=\@M

% allow more spacing between words
\tolerance=\@M

% define new length
\newlength{\contentsmargin}

% set distance from top of page to first float
\setlength{\@fptop}{\z@}

% define an algorithm environment
\DeclareFloatingEnvironment[%
    fileext             = loa,%
    listname            = {List of Algorithms},%
    name                = Algorithm,%
    placement           = htbp,%
    within              = chapter,%
    chapterlistsgaps    = off%
    ]{algorithm}

% figure and table caption names
\renewcommand{\algorithmname}{Algorithm}
\renewcommand{\figurename}{Figure}
\renewcommand{\tablename}{Table}

% caption styling in list of algorithms, figures, and tables
\DeclareCaptionListFormat{algstyle}{\algorithmname\ #2.}
\DeclareCaptionListFormat{figstyle}{\figurename\ #2.}
\DeclareCaptionListFormat{tabstyle}{\tablename\ #2.}

% caption styling inside the text
\captionsetup{justification=centering,labelsep=period,subrefformat=parens}
\captionsetup[algorithm]{aboveskip=12pt,belowskip=20pt,listformat=algstyle}
\captionsetup[figure]{aboveskip=12pt,belowskip=8pt,listformat=figstyle}
\captionsetup[subfigure]{aboveskip=5pt,belowskip=5pt}
\captionsetup[table]{aboveskip=12pt,belowskip=20pt,listformat=tabstyle}

% table cell spacing adjustments
\renewcommand{\theadalign}{cc}
\renewcommand{\theadfont}{\bfseries}
\renewcommand{\theadgape}{\Gape[2pt]}
\renewcommand{\cellalign}{cc}
\renewcommand{\cellgape}{\Gape[2pt]}

% listing settings
\definecolor{listingbg}{rgb}{.95, .95, .95}
\lstset{
    frame               = single,
    keepspaces          = true,
    columns             = flexible,
    basicstyle          = {\footnotesize\ttfamily},
    escapeinside        = {(*<}{>*)},
    backgroundcolor     = \color{listingbg},
    showstringspaces    = false,
    language            = C,
    keywordstyle        = \color{blue},
    stringstyle         = \color{red},
    commentstyle        = \color{teal},
    numbers             = left,
    firstnumber         = 1,
    numberstyle         = \footnotesize\ttfamily,
    numbersep           = 7pt
}

% dotfill command
\def\dotfill{%
    \leavevmode
    \cleaders \hb@xt@ .35em{\hss .\hss}\hfill
    \kern \z@
}

% internal variables
\global\let\@appflag\empty

\global\let\@department\empty
\global\let\@degree\empty
\global\let\@studentnumber\empty

\global\let\@supervisor\empty
\global\let\@supervisoruni\empty
\global\let\@cosupervisori\empty
\global\let\@cosupervisoriuni\empty
\global\let\@cosupervisorii\empty
\global\let\@cosupervisoriiuni\empty
\global\let\@examineri\empty
\global\let\@examineriuni\empty
\global\let\@examinerii\empty
\global\let\@examineriiuni\empty
\global\let\@examineriii\empty
\global\let\@examineriiiuni\empty
\global\let\@examineriv\empty
\global\let\@examinerivuni\empty
\global\let\@examinerv\empty
\global\let\@examinervuni\empty

\global\let\@turkishtitle\empty

% external cover page commands
\newcommand{\department}[1]{%
    \global\def\@department{#1}%
}

\newcommand{\degree}[1]{%
    \global\def\@degree{#1}%
}

\newcommand{\studentnumber}[1]{%
    \global\def\@studentnumber{#1}%
}

\newcommand{\supervisor}[2]{%
    \global\def\@supervisor{#1}%
    \global\def\@supervisoruni{#2}%
}

\newcommand{\cosupervisori}[2]{%
    \global\def\@cosupervisori{#1}%
    \global\def\@cosupervisoriuni{#2}%
}

\newcommand{\cosupervisorii}[2]{%
    \global\def\@cosupervisorii{#1}%
    \global\def\@cosupervisoriiuni{#2}%
}

\newcommand{\examineri}[2]{%
    \global\def\@examineri{#1}%
    \global\def\@examineriuni{#2}%
}

\newcommand{\examinerii}[2]{%
    \global\def\@examinerii{#1}%
    \global\def\@examineriiuni{#2}%
}

\newcommand{\examineriii}[2]{%
    \global\def\@examineriii{#1}%
    \global\def\@examineriiiuni{#2}%
}

\newcommand{\examineriv}[2]{%
    \global\def\@examineriv{#1}%
    \global\def\@examinerivuni{#2}%
}

\newcommand{\examinerv}[2]{%
    \global\def\@examinerv{#1}%
    \global\def\@examinervuni{#2}%
}

\newcommand{\turkishtitle}[1]{%
    \global\def\@turkishtitle{#1}%
}

% helper macro for Thesis/Dissertation
\newcommand{\thesisdissertation}{%
    \ifx\@degree\empty
        Thesis%
    \else
        \def\temp{Doctor of Philosophy}%
        \ifx\@degree\temp
            Dissertation%
        \else
            Thesis%
        \fi
    \fi
}

% internal commands
\newcommand{\@MakeThesisTitle}{%
    \begin{titlepage}
    \begin{center}
    \includegraphics[width=2.5cm]{figures/logo.png}\\
    YEDITEPE UNIVERSITY\\
    GRADUATE SCHOOL%
    \vskip 12ex
    \MakeUppercase{\@title}%
    \vskip 12ex
    A \thesisdissertation\ Submitted\\

    by\\
    \@author%
    \vfill
    In Partial Fulfillment\\
    of the Requirements for the Degree of\\
    \ifx\@degree\empty
        \@@warning{No \string\degree\space given}
    \else
        \@degree\\
    \fi
    in\\
    \ifx\@department\empty
        \@@warning{No \string\department\space given}
    \else
        \@department%
    \fi
    \vfill
    Supervisor\\
    \@supervisor%
    \vskip 12ex
    Istanbul - \@date%
    \vskip 5ex
    \end{center}
    \end{titlepage}
}

\newcommand{\@MakeReportTitle}{%
    \begin{titlepage}
    \begin{center}
    \includegraphics[width=2.5cm]{figures/logo.png}\\
    YEDITEPE UNIVERSITY\\
    GRADUATE SCHOOL%
    \vfill
    \ifx\@ptype\tw@
        THESIS PROPOSAL\\
    \fi
    \ifx\@ptype\thr@@
        THESIS FOLLOW-UP REPORT\\
    \fi
    \vfill
    \@author\\
    \ifx\@department\empty
        \@@warning{No \string\department\space given}
    \else
        \@department%
    \fi
    \vfill
    \@date%
    \vskip 5ex
    \end{center}
    \end{titlepage}
}

\newcommand{\@MakeApprovalPage}{%
    \addtocounter{page}{1}
    \begin{center}
    \MakeUppercase{\@title}\\[8ex]
    by\\
    \@author
    \end{center}
    \null\vfill
    Approved by:\\[2ex]
    \ifx\@supervisor\empty
        \ifx\@cosupervisori\empty
            \ifx\@cosupervisorii\empty
                \@@warning{No \string\supervisor\space given}
            \fi
        \fi
    \else
        \@supervisor\\
        (\@supervisoruni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\
        (Thesis Supervisor)\\[3ex]
    \fi
    \ifx\@cosupervisori\empty
    \else
        \@cosupervisori\\
        (\@cosupervisoriuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\
        (Thesis Co-Supervisor)\\[3ex]
    \fi
    \ifx\@cosupervisorii\empty
    \else
        \@cosupervisorii\\
        (\@cosupervisoriiuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\
        (Thesis Co-Supervisor)\\[3ex]
    \fi
    \ifx\@examineri\empty
    \else
        \@examineri\\
        (\@examineriuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\[7ex]
    \fi
    \ifx\@examinerii\empty
    \else
        \@examinerii\\
        (\@examineriiuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\[7ex]
    \fi
    \ifx\@examineriii\empty
    \else
        \@examineriii\\
        (\@examineriiiuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\[7ex]
    \fi
    \ifx\@examineriv\empty
    \else
        \@examineriv\\
        (\@examinerivuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}\\[7ex]
    \fi
    \ifx\@examinerv\empty
    \else
        \@examinerv\\
        (\@examinervuni) \hfill \makebox[6cm]{\dotfill \hskip 1cm}
    \fi
    \null\vfill
    DATE OF APPROVAL:\hskip 1em\hb@xt@ 1.5em{\dotfill}\ /%
    \ \hb@xt@ 1.5em{\dotfill}\ /\ 20\hb@xt@ 1.5em{\dotfill}%
}

\newcommand{\@MakeCommitteePage}{%
    \addtocounter{page}{1}
    \begin{center}
    \MakeUppercase{\@title}%
    \end{center}
    \vfill
    STEERING COMMITTEE:
    \vskip 7.5ex
    \begin{tabbing}
    \hspace{3cm} \= \hspace{2cm} \= \kill
    \ifx\@supervisor\empty
        \ifx\@cosupervisori\empty
            \ifx\@cosupervisorii\empty
                \@@warning{No \string\supervisor\space given}
            \fi
        \fi
    \else
        Thesis Advisor \> : \> \@supervisor \\
        \> \> \@supervisoruni \\
        \\
    \fi
    \ifx\@cosupervisori\empty
    \else
        Co-Advisor \> : \> \@cosupervisori \\
        \> \> \@cosupervisoriuni \\
        \\
    \fi
    \ifx\@cosupervisorii\empty
    \else
        Co-Advisor \> : \> \@cosupervisorii \\
        \> \> \@cosupervisoriiuni \\
        \\
    \fi
    \ifx\@examineri\empty
    \else
        Member \> : \> \@examineri \\
        \> \> \@examineriuni \\
        \\
    \fi
    \ifx\@examinerii\empty
    \else
        Member \> : \> \@examinerii \\
        \> \> \@examineriiuni \\
        \\
    \fi
    \ifx\@examineriii\empty
    \else
        Member \> : \> \@examineriii \\
        \> \> \@examineriiiuni \\
        \\
    \fi
    \ifx\@examineriv\empty
    \else
        Member \> : \> \@examineriv \\
        \> \> \@examinerivuni \\
        \\
    \fi
    \ifx\@examinerv\empty
    \else
        Member \> : \> \@examinerv \\
        \> \> \@examinervuni \\
    \fi
    \end{tabbing}
    \vfill\null
}

\newcommand{\@MakeDeclarationPage}{%
    \chapter*{Declaration of Originality}
    I hereby declare that this thesis is my own work and that all information
    in this thesis has been obtained and presented following academic rules
    and ethical conduct. I have fully cited and referenced all material and
    results as required by these rules and conduct, and this thesis study
    does not contain any plagiarism. The necessary permissions have been
    obtained if any material used in the thesis requires copyright. No
    material from this thesis has been used to award another degree.

    To the best of my knowledge and belief, it contains no material
    previously published or written by another person nor material accepted
    for the award of any other degree except where due acknowledgment has
    been made in the text.

    I accept all kinds of legal liability that may arise in cases contrary
    to these situations.
    \vskip 12ex
    \begin{flushright}
    \begin{tabular}{c}
    \@author\\[5ex]
    \makebox[5cm]{\dotfill}
    \end{tabular}\hspace{.5cm}\null
    \end{flushright}
}

\def\@beginabb{%
\begin{tabbing}
    \hspace{4cm} \= \kill
}

\def\@closeabb{%
    \end{tabbing}
}

\def\@abbline#1{
    \parbox[t]{11.4cm}{#1\par\vspace{4pt}}
}

% external commands
\newcommand{\MakeFrontMatter}{%
    \ifx\@dtype\@ne
        \@MakeThesisTitle
        \@MakeApprovalPage
        \@MakeDeclarationPage
        \input{frontmatter/abstract-en}
        \input{frontmatter/abstract-tr}
        \IfFileExists{frontmatter/dedication}%
            {\input{frontmatter/dedication}}%
            {}%
        \IfFileExists{frontmatter/acknowledgements}%
            {\input{frontmatter/acknowledgements}}%
            {}%
        \tableofcontents
%        \listofalgorithms
        \listoffigures
        \listoftables
        \input{frontmatter/abbreviations}
    \fi
    \ifx\@dtype\tw@
        \@MakeReportTitle
        \@MakeCommitteePage
        \tableofcontents
%        \listofalgorithms
%        \listoffigures
%        \listoftables
%        \input{frontmatter/abbreviations}
    \fi
}

\newcommand{\abbline}[2]{#1 \> \@abbline{#2} \\}

\newcommand{\MakeReferences}{%
    \bibliographystyle{vancouver}
    \urlstyle{same}
    \bibliography{references}\label{chp:ref}
}

% chapter names
\def\abstractname{Abstract}
\def\turkishabstractname{Özet}
\def\dedicationname{Dedication}
\def\acknowledgementname{Acknowledgements}
\def\contentsname{Table of Contents}
\def\listfigurename{List of Figures}
\def\listtablename{List of Tables}
\def\listalgorithmname{List of Algorithms}
\def\listabbreviationname{List of Abbreviations}
\def\bibname{References}
\def\appendixname{Appendix}

% external environments
\renewenvironment{abstract}{%
    \chapter*{\abstractname}
    {%
        \centering
        \normalfont \interlinepenalty \@M
        \normalsize \bfseries
        \MakeUppercase{\@title}\par\nobreak
        \vskip 14pt
    }%
}{\par\vfill\null}

\newenvironment{turkishabstract}{%
    \chapter*{\turkishabstractname}
    \ifx\@turkishtitle\empty
        \@@warning{No \string\turkishtitle\space given}
    \else
    {%
        \uccode`\i='235%
        \lccode`\I='031%
        \centering
        \normalfont \interlinepenalty \@M
        \normalsize \bfseries
        \MakeUppercase{\@turkishtitle}\par\nobreak
        \vskip 14pt
    }%
    \fi
}{\par\vfill\null}

\newenvironment{dedication}{%
    \chapter*{\dedicationname}
}{\par\vfill\null}

\newenvironment{acknowledgements}{%
    \chapter*{\acknowledgementname}
}{\par\vfill\null}

\newenvironment{abbreviations}{%
    \chapter*{\listabbreviationname}
    \@beginabb
}{\@closeabb}

% section and toc depth settings
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

% dottedtocline redefinition
\def\@dottedtocline#1#2#3#4#5{%
    \ifnum #1>\c@tocdepth \else
    {%
        \leftskip #2\relax \rightskip 2.5em\relax \parfillskip -\rightskip
        \parindent #2\relax \@afterindenttrue
        \interlinepenalty \@M
        \leavevmode
        \@tempdima #3\relax
        \ifnum #1=-10 \else
            \advance \leftskip \@tempdima \null \nobreak \hskip -\leftskip
        \fi
        {#4}\nobreak
        \setlength{\contentsmargin}{2.5em - \widthof{#5}}%
        \leaders \hb@xt@ .35em{\hss .\hss}\hskip \the\contentsmargin plus 1fill%
        \nobreak #5\par \ifnum #1=-10 \vskip 10\p@\relax \fi
    }%
    \fi
}

% table of contents dotted line settings
\renewcommand*\l@chapter{\@dottedtocline{0}{0em}{2em}}
\renewcommand*\l@section{\@dottedtocline{1}{1em}{2.75em}}
\renewcommand*\l@subsection{\@dottedtocline{2}{2.75em}{3.5em}}
\renewcommand*\l@subsubsection{\@dottedtocline{3}{5.25em}{4em}}
\renewcommand*\l@paragraph{\@dottedtocline{4}{8.5em}{4.75em}}

% list of algorithms, figures, and tables dotted line settings
\renewcommand*\l@algorithm{\@dottedtocline{-10}{0em}{7em}}
\renewcommand*\l@figure{\@dottedtocline{-10}{0em}{5.35em}}
\renewcommand*\l@table{\@dottedtocline{-10}{0em}{5em}}

% settings related to chapter, section, subsection, etc.
\renewcommand{\chapter}{%
    \clearpage
    \global\@topnum\z@
    \@afterindenttrue
    \secdef\@chapter\@schapter
}

\def\@chapter[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}%
        \ifx\@appflag\empty
            \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter.}%
                \texorpdfstring{\MakeUppercase{#1}}{#1}}%
        \else
            \addcontentsline{toc}{chapter}{\texorpdfstring{\MakeUppercase{%
                \appendixname}\ \thechapter}{\appendixname\ \thechapter}}%
        \fi
    \fi
    \chaptermark{#1}%
    \@makechapterhead{#2}%
    \@afterheading
}

\def\@makechapterhead#1{{%
    \parindent \z@ \normalfont
    \interlinepenalty \@M
    \large \bfseries
    \ifnum \c@secnumdepth >\m@ne
        \ifx\@appflag\empty
            \@hangfrom{\thechapter .\hskip .5em\relax}%
        \else
            \MakeUppercase{\appendixname} \thechapter :\hskip .5em\relax
        \fi
    \fi
    \MakeUppercase{#1}\par\nobreak
    \vskip 14\p@
}}

\def\@schapter#1{%
    \addcontentsline{toc}{chapter}{\texorpdfstring{\MakeUppercase{#1}}{#1}}%
    \@makeschapterhead{#1}%
    \@afterheading
}

\def\@makeschapterhead#1{{%
    \def\@arga{#1}%
    \def\@argb{\bibname}%
    \ifx\@arga\@argb \else
        \centering
    \fi
    \parindent \z@ \normalfont
    \interlinepenalty \@M
    \large \bfseries
    \MakeUppercase{#1}\par\nobreak
    \vskip 14\p@
}}

\def\@sect#1#2#3#4#5#6[#7]#8{%
    \ifnum #2>\c@secnumdepth
        \let\@svsec\@empty
    \else
        \refstepcounter{#1}%
        \protected@edef\@svsec{\csname the#1\endcsname.\hskip .5em\relax}%
    \fi
    \@tempskipa #5\relax
    \ifdim \@tempskipa >\z@
        \begingroup
        #6%
        {%
            \@hangfrom{\hskip #3\relax\@svsec}%
            \interlinepenalty \@M
            \ifnum #2=1 \MakeUppercase{#8}\else #8\fi
            \@@par
        }%
        \endgroup
        \csname #1mark\endcsname{#7}%
        \addcontentsline{toc}{#1}{%
            \ifnum #2>\c@secnumdepth \else
                \protect\numberline{\csname the#1\endcsname.}%
            \fi
            \ifnum #2=1 \texorpdfstring{\MakeUppercase{#7}}{#7}%
            \else #7%
            \fi}%
    \else
        \def\@svsechd{%
            #6%
            {%
                \hskip #3\relax\@svsec
                \ifnum #2=1 \MakeUppercase{#8}\else #8\fi
            }%
            \csname #1mark\endcsname{#7}%
            \addcontentsline{toc}{#1}{%
                \ifnum #2>\c@secnumdepth \else
                    \protect\numberline{\csname the#1\endcsname.}%
                \fi
                \ifnum #2=1 \texorpdfstring{\MakeUppercase{#7}}{#7}%
                \else #7%
                \fi}%
        }%
    \fi
    \@xsect{#5}%
}

\renewcommand{\section}{%
    \@startsection{section}%
        {1}%
        {\z@}%
        {24\p@}%
        {24\p@}%
        {\normalfont\normalsize\bfseries}%
}

\renewcommand{\subsection}{%
    \@startsection{subsection}%
        {2}%
        {\z@}%
        {24\p@}%
        {24\p@}%
        {\normalfont\normalsize\bfseries}%
}

\renewcommand{\subsubsection}{%
    \@startsection{subsubsection}%
        {3}%
        {\z@}%
        {10\p@}%
        {10\p@}%
        {\normalfont\normalsize\bfseries\itshape}%
}

\renewcommand{\paragraph}{%
    \@startsection{paragraph}%
        {4}%
        {\z@}%
        {10\p@}%
        {10\p@}%
        {\normalfont\normalsize\bfseries}%
}

% bibliography settings
\def\@biblabel#1{#1.\hfill}

\def\@citex[#1]#2{%
    \leavevmode
    \let\@citea\@empty
    \@cite{%
        \@for\@citeb:=#2\do{%
            \@citea\def\@citea{,\penalty\@M}%
            \edef\@citeb{\expandafter\@firstofone\@citeb\@empty}%
            \if@filesw\immediate\write\@auxout{\string\citation{\@citeb}}\fi
            \@ifundefined{b@\@citeb}{\hbox{\reset@font\bfseries ?}%
            \G@refundefinedtrue
            \@latex@warning
                {Citation `\@citeb' on page \thepage \space undefined}}%
                {\@cite@ofmt{\csname b@\@citeb\endcsname}}%
        }%
    }{#1}%
}
      
% appendix settings
\renewcommand{\appendix}{%
    \par
    \setcounter{chapter}{0}%
    \setcounter{section}{0}%
    \global\def\@chapapp{\appendixname}%
    \global\def\thechapter{\@Alph\c@chapter}%
    \let\@appflag\@ne
}

% footnote numbers does not reset each chapter
\counterwithout{footnote}{chapter}
\setcounter{footnote}{0}

% footnote rule adjustments
\renewcommand{\footnoterule}{%
    \kern -3\p@
    \hrule \@width 5cm \@height .4\p@
    \kern 2.6\p@
}

% footnote text adjustments
\renewcommand{\@makefntext}[1]{%
    \parindent \z@%
    \noindent \hb@xt@ 1cm{\hss\@makefnmark\ }#1%
}

% the size required in quotations is 10pt
% adjusted 1cm spacing from left and right margins
\renewenvironment{quote}{
    \normalfont \footnotesize
    \list{}{\rightmargin=1cm \leftmargin=1cm}
    \item \relax
}{\endlist}

% hyperref & cleveref must be loaded at the very end
\RequirePackage[hidelinks,linktoc=all]{hyperref}
\RequirePackage[capitalize,nameinlink,noabbrev]{cleveref}

%%
%% End of "thesis.cls" file
%%
