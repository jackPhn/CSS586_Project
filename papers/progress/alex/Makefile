.PHONY: watch

default: progress-alex.pdf

progress-alex.pdf: progress-alex.tex progress-alex.bib
	pdflatex progress-alex
	bibtex progress-alex
	pdflatex progress-alex
	pdflatex progress-alex

watch:
	ladtexmk -pdf -pvc paper.tex