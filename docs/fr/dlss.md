# NVIDIA DLSS dans le lanceur

[English](../dlss.md) · [Guide du lanceur](lanceur.md)

**Super Resolution** reconstruit l’image affichée à partir d’un rendu de résolution inférieure, de la profondeur et des vecteurs de mouvement. Le mode Quality conserve davantage de détails. Balanced, Performance et Ultra Performance réduisent progressivement le nombre de pixels calculés. DLAA applique l’anticrénelage à la résolution native de sortie.

**Frame Generation 2x** utilise le mouvement entre deux images rendues pour insérer une image intermédiaire. Le jeu continue de lire les commandes au rythme des images rendues. Le **27 novembre 2026**, une mesure avec la **build VIctory-NVIDIA** sur une **GeForce RTX 4070 SUPER** a donné **12 images rendues et 24 affichées par seconde**. Le compteur de VIctory indique les images rendues. Le journal de Streamline compte les images affichées.

## Installer Streamline

1. Utilisez une GeForce RTX avec un pilote NVIDIA récent. La génération 2x nécessite une RTX série 40 ou 50. Dans Windows, ouvrez **Paramètres → Système → Affichage → Graphiques**, activez la **planification de processeur graphique à accélération matérielle** (HAGS) dans les paramètres graphiques par défaut, puis redémarrez le PC.
2. Dans **Play → NVIDIA DLSS**, cliquez sur **Télécharger les fichiers NVIDIA**. Le bouton ouvre la page des versions de NVIDIA dans le navigateur. Téléchargez vous-même l’archive du SDK Streamline (version de référence **2.14.16**) et décompressez-la hors du dépôt. NVIDIA fournit les DLL sous ses propres licences. AnyXboxX n’en distribue aucune. [Versions officielles de NVIDIA Streamline](https://github.com/NVIDIA-RTX/Streamline/releases).
3. Dans **Streamline folder**, sélectionnez le dossier `bin\x64` de la version extraite. Gardez ensemble `sl.interposer.dll`, `sl.common.dll`, `sl.dlss.dll`, `sl.dlss_g.dll`, `sl.reflex.dll`, `sl.pcl.dll`, `nvngx_dlss.dll` et `nvngx_dlssg.dll`. Le lanceur vérifie la signature Authenticode de `sl.interposer.dll` et exige le signataire **NVIDIA Corporation**. En cas de refus, vérifiez le dossier choisi et utilisez les fichiers de la version publiée par NVIDIA.
4. Activez **DLSS Super Resolution** et choisissez un mode, activez **Frame Generation 2x**, ou combinez les deux. Si la génération est grisée, le lanceur indique le prérequis manquant. En 1440p, activez les deux options car le traitement FidelityFX du jeu, utilisé avec la génération seule, accepte un nombre limité de résolutions.

## Réglages et limites

Le lanceur conserve ces choix dans `%LOCALAPPDATA%\AnyXboxX\launcher\settings.toml`. Chaque fonction sélectionne le chemin FidelityFX du jeu (`VICTORY_UPSCALER=7`) et utilise `VICTORY_STREAMLINE_DIR`. Super Resolution règle `VICTORY_DLSS=1`, `VICTORY_DLSS_MODE` et `VICTORY_DLSS_PRESET`. La génération règle `VICTORY_FRAMEGEN=1`, `VICTORY_THIN=1` et `VICTORY_FRAMEGEN_HUDLESS`. Les valeurs saisies dans **Advanced** ont priorité. Au lancement, la vérification de signature porte sur le dossier effectivement transmis.

- **Latence.** Pour générer l’image intermédiaire, une image rendue est retenue. À 12 images rendues par seconde, le jeu lit les commandes environ toutes les 83 ms et la mise en mémoire tampon peut ajouter du délai.
- **Focus de la fenêtre.** La génération se suspend quand la fenêtre du jeu perd le focus. Ramenez-la au premier plan avant de comparer les deux compteurs.
- **Interface.** Laissez **Exclude the HUD from generated motion** activé. Ce réglage utilise une copie de la scène prise avant l’affichage de l’interface. Les éléments dessinés séparément peuvent encore se déformer au passage d’un objet en mouvement. Pour comparer, réglez `VICTORY_FRAMEGEN_HUDLESS=0` dans Advanced.
