import time


class TrainingTimer:
    """
    Chronomètre la durée d'un entraînement.

    Supporte une durée initiale (cumulée) pour la reprise
    d'un entraînement après un crash.
    """

    def __init__(self, initial_duration=0.0):
        """
        initial_duration : durée cumulée initiale en secondes
                           (0.0 par défaut pour un nouvel entraînement)
        """
        self.initial_duration = float(initial_duration)

        self.start_time = None
        self.end_time = None

    def start(self):
        """
        Démarre le chronomètre.
        """
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self):
        """
        Arrête le chronomètre et retourne la durée TOTALE en secondes.

        La durée retournée inclut :
        - La durée initiale (cumulée des runs précédents)
        - La durée du run actuel

        Retourne :
            float : durée totale en secondes
        """
        if self.start_time is None:
            raise RuntimeError("Le chronomètre n'a pas été démarré.")

        self.end_time = time.perf_counter()

        elapsed = self.end_time - self.start_time

        return self.initial_duration + elapsed

    def elapsed(self):
        """
        Retourne la durée écoulée depuis le démarrage
        SANS arrêter le chronomètre.

        La durée inclut :
        - La durée initiale
        - La durée écoulée depuis start()

        Retourne :
            float : durée totale en secondes
        """
        if self.start_time is None:
            return self.initial_duration

        elapsed = time.perf_counter() - self.start_time

        return self.initial_duration + elapsed

    @staticmethod
    def format_duration(duration_seconds):
        """
        Convertit une durée en format lisible.

        Format : "XXh XXmin XX.XXs"
        """

        hours = int(duration_seconds // 3600)

        minutes = int((duration_seconds % 3600) // 60)

        seconds = (duration_seconds % 60)

        return (f"{hours:02d}h " f"{minutes:02d}min " f"{seconds:05.2f}s")
