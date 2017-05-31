(TeX-add-style-hook
 "hmcmc_update"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("geometry" "margin=1in") ("caption" "font=footnotesize" "labelfont=bf") ("color" "usenames" "dvipsnames") ("graphicx" "draft") ("biblatex" "citestyle=authoryear" "bibstyle=authoryear" "backend=biber")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "inputenc"
    "geometry"
    "amsmath"
    "amsthm"
    "amsfonts"
    "wrapfig"
    "lipsum"
    "booktabs"
    "ragged2e"
    "amssymb"
    "caption"
    "subcaption"
    "color"
    "graphicx"
    "float"
    "hhline"
    "tabularx"
    "biblatex")
   (TeX-add-symbols
    "rojo"
    "azul")
   (LaTeX-add-environments
    "theorem")
   (LaTeX-add-bibliographies
    "ref.bib")))

