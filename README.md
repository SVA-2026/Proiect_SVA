**# Proiect_SVA
Multi‑View Consistency and Simple 3D Reconstruction**

Am implementat prima etapa pentru reconstructia 3D a unui obiect dintr-un set de imagini 2D. 
In aceasta faza ne-am ocupat de detectia trasaturilor, potrivirea lor intre vederi si estimarea liniilor epipolare.

**1. Date folosite**
Am ales 5 imagini in care obiectul de interes este o masina mica de pompieri ce faciliteaza detectia de puncte cheie. Imaginile au fost ordonate pentru a avea intre doua imagini vecine si puncte comune.
Imaginile au fost redimensionate la o latime de 800px pentru o vizualizare corecta pe ecran (inainte de redimensionare, vizualizarea nu era clara si nu se puteau vizualiza punctele de interes)

**2. Arhitectura proiectului**
Am ales sa abordan o organizare ce respecta SOLID pentru o mai usoara observare a functionarii si pentru o arhitectura mai clara. 
Modulele au fost implementate separat pentru incarcare date (loaders), procesare (features), partea de matematica (geometry) si vizualizare rezultate (vizualization ).
Astfel, prin aceasta abordare, proiectul nostru poate fi schimbat si actualizat in orice moment, fara a strica tot codul.

**3. Logica si Functionare**
 Am ales sa folosim algoritmul SIFT (Scale-Invariant Feature Transform) pentru a extrage punctele de interes ce sunt invariante la rotatie, scara si schimbari de iluminare.

Pentru potrivirea perechilor: am folosit FLANN (Fast Library for Approximate Nearest Neighbors) pentru a putea cauta rapid in descriptorii nostri.
De asemenea, am ales sa folosim Lowe's Ratio Test pentru eliminarea potrivirilor ambigue ( unde distanta dintre cei mai buni 2 candidati este prea mica ).

--> Estimarea Matricei Fundamentale (F): se calculeaza relatia geometrica dintre doua imagini.
--> RANSAC: acesta alege doar punctele ce sunt din aceeasi categorie si se ignora liniile ce nu sunt conforme categoriilor predominante. (ex. Daca majoritatea liniilor merg pe orizonata, dar doar 2-3 pe verticala, acestea vor fi ignorate, fiind considerate incorecte ).

