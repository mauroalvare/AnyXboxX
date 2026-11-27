# Versions bêta Windows x64

Version anglaise dans [docs/beta-releases.md](../beta-releases.md).

1. Intégrez les optimisations voulues dans **`beta/optimizations`** sur [le fork](https://github.com/mauroalvare/AnyXboxX/tree/beta/optimizations). Le workflow récupère cette branche et la branche `main` du dépôt d’origine au début de chaque exécution.
2. Sur le fork, ouvrez **Actions → Windows x64 beta → Run workflow**, sélectionnez `main` et lancez le workflow. Chaque exécution crée sa propre branche `beta/integration-…` et compile et teste cette révision précise.
3. Ouvrez [Releases](https://github.com/mauroalvare/AnyXboxX/releases) après le succès du workflow. Dans **Assets** de la nouvelle préversion `beta-…`, téléchargez `AnyXboxX-windows-x64.zip`. Les notes de publication et le fichier `revisions.json` dans le ZIP donnent les deux SHA d’origine et le commit testé.
4. Extrayez le ZIP, lancez `Play-AnyXboxX.cmd` et choisissez le dossier de votre propre copie du jeu. Consultez les [prérequis Windows](../running-gta6.md#requirements).

Si l’intégration échoue, ouvrez **Summary** dans l’exécution Actions pour trouver les fichiers en conflit. Fusionnez la branche `main` du dépôt d’origine dans `beta/optimizations` dans votre copie locale, résolvez les conflits, poussez le résultat et lancez une nouvelle exécution.

GitHub peut refuser la création de la branche d’intégration si le dépôt d’origine introduit des fichiers de workflow inédits dans le fork. Dans ce cas, configurez le secret Actions `BETA_INTEGRATION_TOKEN` du fork avec un jeton à permissions fines limité à ce fork, en écriture sur **Contents** et **Workflows**. La publication utilise le jeton Actions.
