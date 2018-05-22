# 1. domača naloga: PCA in potenčna metoda

## Navodila za oddajo

Implementiraj metodo glavnih komponent (PCA) in jo uporabi na podatkih o števkah (train.csv na portalu Kaggle in tekmovanju Digit Recognizer https://www.kaggle.com/c/digit-recognizer/data, preden prevzameš datoteko se boš moral na ta portal prijaviti). Pri implementaciji vseh treh spodaj naštetih razredov se zgleduj po razredu sklearn.decomposition.PCA http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html, tako da implementiraš vsaj metode fit in transform ter za shranjevanje lastnih vektorjev in lastnih vrednosti uporabiš atribute components_, explained_variance_ in explained_variance_ratio_. Povprečne vrednosti atributov, ki jih boš sicer uporabil za centriranje učnih podatkov, shrani v atribut razreda mean_; ta atribut bo uporaben v metodi transform_, saj moraš pred PCA transformacijo poljubnih podatkov (torej, ni rečeno samo testnih) vhodno matriko centrirati. Implementacijo preveri na regresijskih testih https://github.com/BlazZupan/ozp/blob/master/tests/pca-test.py. Poleg kode (ena datoteka) oddaj sliko s projekcijo podatkov v dvodimenzionalni prostor, ki ga določata prvi dve komponenti.

Tipične operacije, ki jih naj podpirajo vsi trije implementirani razredi, so razvidne iz spodnje kode:

```
iris = datasets.load_iris()
X = iris.data
train, test = X[:145], X[145:]

my_pca = PowerPCA(n_comp)
my_pca.fit(train)
Z = my_pca.transform(test)
```
1. V razredu **EigenPCA** implementiraj metodo PCA tako, da lastne vektorje kovariančne matrike in pripadajoče lastne vrednosti v metodi fit izračunaš s pomočjo funkcije numpy.linalg.eigh. [10 točk]
2. V razredu **PowerPCA** implementiraj iterativno potenčno tehniko za izračun lastnega vektorja kovariančne matrike. Metodo uporabi na centriranih podatkih in njihovi kovariančni matriki, izračunaj prvo komponento PCA, nato izračunaj novo kovariančno matriko za podatke, ki jim odšteješ projekcije na prvo komponento, in postopek ponovi tako, da izračunaš še drugo komponento. Preveri delovanje tvoje implementacije na poljubnih podatkih (priporočam na primer kakšne, zgrajene na roko, ali pa podatkovni nabor Iris). Na koncu delovanje kode preveri še na regresijskih testih. Potenčno metodo ustavi, kadar je Frobeniusova norma razlike lastnih vektorjev pridobljenih v trenutni in prejšnji iteraciji manjša od neke konstante (recimo, eps=1e-10). Kot dodaten parameter te metode lahko uporabiš maksimalno število iteracij (na primer, 1000). [20 točk]. 
3. Podobno kot zgoraj, tokrat v ločenem razredu **OrtoPCA** potenčno metodo uporabiš samo enkrat in jo kombiniraš z Gram Schmidtovo ortogonalizacijo https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process (kodo za to spiši sam). Ustavitveni kriteij naj bo implemntiran podobno kot v prejšnjih nalogi. [20 točk]
4. Izriši projekcijo podatkov tako, da vsak primer predstaviš s točko na razsevnem grafu s koordinatama ki ustrezata prvima dvema komponentama. Barva točke na ustreza razredu (števka). Graf dopolni tako, da v centru točk, ki pripadajo isti števki, izpišeš ustrezno številko. [10 točk]
Preostali del ocene sestavlja ocena sloga programiranja [10 točk] in sodelovanje pri ocenjevanju (10% ocene). Pri slogu programiranje bomo upoštevali kompaktnost kode in upoštevanje priporočil PEP 8. Za ocenjevanje boš naprošen, da oceniš domače naloge treh sošolk oz. sošolcev. Nekatere naloge bomo ocenili skupaj v razredu.

Dodatek: Katerakoli druga metoda, npr. razcep Choleskega ali pa QR razcep, ali pa povzporedenje kakšne od zgornjih metod. Za to metodo spišite regresijske teste. [10 točk] Dodatne točke [20 točk], če pokažeš, da je ta metoda bistveno hitrejša kot implementacija v sklearn (bistveno=čas izvajanja se vsaj prepolovi). To novo metodo implementiraj v svojem razredu, čas pa izmeri v zadnjem delu tvoje skripte, k se prične z 

```
if __name__ == "__main__":
```
**Oddaja**. Oddajte kodo (pca.py) ter sliko s projekcijo podatkov (pca.pdf). V dokumentih, ki jih oddajate, naj ne bo nikjer zaznamka z vašim imenom ali vpisno številko, saj bodo tokrat naloge ocenjevali vaši kolegi in bi se radi želeli izogniti morebitnim pristranostim.