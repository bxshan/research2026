$bibtex_use = 2;
$pdflatex = 'pdflatex -synctex=1 %O %S';

# Style/bib files live in sty/; add it to the search paths (// = recursive).
$ENV{'TEXINPUTS'} = './sty//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = './sty//:' . ($ENV{'BSTINPUTS'} // '');
