#!/bin/bash
xname=1
yname=0
mkdir letters
cp Sloan.otf letters
for Y in 0.05 0.06 0.075 0.1 0.125 0.15 0.2 0.25 0.35 0.5 1
do
  sed s/ratio\{1\}/ratio\{$Y\}/g $1.tex > letters/$yname\_.tex
  for X in C D E F L N O P T Z
  do
    sed s/selectfont\ E/selectfont\ $X/g letters/$yname\_.tex > letters/$xname\_$yname\_.tex
    cd letters
    xelatex -synctex=1 -interaction=nonstopmode $xname\_$yname\_.tex
    cd ..
    rm letters/$xname\_$yname\_.tex
    rm letters/$xname\_$yname\_.log
    rm letters/$xname\_$yname\_.aux
    rm letters/$xname\_$yname\_.synctex.gz
    echo $((xname++))
  done
  rm letters/$yname\_.tex
  echo $((yname++))
  xname=1
done

rm letters/Sloan.otf
zip -r letters.zip letters
rm -r letters



