default: paper.pdf

paper.pdf: paper.tex paper.bib
	latexmk -bibtex -pdf paper

watch:
	latexmk -bibtex -pdf -pvc paper
