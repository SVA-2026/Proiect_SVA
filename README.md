**# Proiect_SVA
Multi‑View Consistency and Simple 3D Reconstruction**

Am implementat prima etapă pentru reconstrucția 3D a unui obiect dintr-un set de imagini 2D. 
În aceasta fază ne-am ocupat de detecția trăsăturilor, potrivirea lor între vederi și estimarea liniilor epipolare.

**1. Date folosite**

Am ales 5 imagini în care obiectul de interes este o mașnă mică de pompieri ce facilitează detecția de puncte cheie. Imaginile au fost ordonate pentru a avea între două imagini vecine și puncte comune.
Imaginile au fost redimensionate la o lățime de 800px pentru o vizualizare corectă pe ecran (înainte de redimensionare, vizualizarea nu era clară și nu se puteau vizualiza punctele de interes). Am realizat și o conversie, deoarece procesarea se realizează pe variantele grayscale ale imaginilor pentru a extrage gradienții de intensitate necesari algoritmului SIFT.

**2. Arhitectura proiectului**

Am ales să abordăm o organizare ce respectă SOLID pentru o mai ușoară observare a funcționării și pentru o arhitectura mai clară. 
Modulele au fost implementate separat pentru încarcare date (loaders), procesare (features), partea de matematică (geometry) și vizualizare rezultate (vizualization).
Astfel, prin această abordare, proiectul nostru poate fi schimbat și actualizat în orice moment, fără a strica tot codul.

**3. Logică și Funcționare**

 Am ales să folosim algoritmul SIFT (Scale-Invariant Feature Transform) pentru a extrage punctele de interes ce sunt invariante la rotație, scară si schimbări de iluminare. Distingem în această etapă variabilele keypoints (kp), ce reprezintă locațiile (x,y) ale trăsăturile distinctive, și descriptors (des), care sunt vectori de 128 de valori care descriu numeric aspectul vizual al fiecărui punct.

Pentru potrivirea perechilor: am folosit FLANN (Fast Library for Approximate Nearest Neighbors) pentru a putea căuta rapid în descriptorii noștri.
De asemenea, am ales să folosim Lowe's Ratio Test pentru eliminarea potrivirilor ambigue (unde distanța dintre cei mai buni 2 candidați este prea mică) și avem un prag setat la 0.75.

--> Estimarea Matricei Fundamentale (F): se calculează relația geometrică dintre două imagini.

--> RANSAC: acesta alege doar punctele ce sunt din aceeași categorie și se ignoră liniile ce nu sunt conforme categoriilor predominante. (ex. Daca majoritatea liniilor merg pe orizonatală, dar doar 2-3 pe verticală, acestea vor fi ignorate, fiind considerate incorecte).

Ca și flow de lucru distingem 5 puncte cheie:
1. Încărcăm imaginile + redimensionare + grayscale.
2. Extragem trăsăturile cu SIFT.
3. Realizăm matching-ul cu FLANN.
4. Estimăm geometria.
5. Vizualizăm rezultatele finale, adică inliners.

Ca și rezultate avem imaginile cu matching-ul + output-ul din terminal în ceea ce privește matricea F și punctele comune ale imaginilor.

<img width="1906" height="786" alt="image" src="https://github.com/user-attachments/assets/d8cd2907-2814-46bc-9f5c-1d23a90c2673" />


<img width="676" height="310" alt="image" src="https://github.com/user-attachments/assets/267eff97-fc12-4cae-955d-40ca97f4548c" />
