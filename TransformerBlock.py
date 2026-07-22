import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    """
    tête d'auto-attention.

    Calcule les projections clé / requête / valeur puis applique une
    attention pondérée où chaque position ne peut regarder que les
    positions précédentes (masque triangulaire inférieur), ce qui rend
    cette tête adaptée à la génération autorégressive de type GPT.
    """

    def __init__(self, dimension_embedding, taille_bloc, taille_tete):
        """
        Args:
            dimension_embedding (int): dimension des vecteurs d'entrée.
            taille_bloc (int): longueur maximale de la séquence (contexte).
            taille_tete (int): dimension de sortie de cette tête d'attention.
        """
        super().__init__()
        self.cle = nn.Linear(dimension_embedding, taille_tete, bias=False)
        self.requete = nn.Linear(dimension_embedding, taille_tete, bias=False)
        self.valeur = nn.Linear(dimension_embedding, taille_tete, bias=False)
        self.register_buffer('triangle_inferieur', torch.tril(torch.ones(taille_bloc, taille_bloc)))

    def forward(self, x):
        """
        Calcule la sortie de la tête d'attention pour un batch de séquences.

        Args:
            x (torch.Tensor): tenseur d'entrée de forme (batch, temps, canaux).

        Returns:
            torch.Tensor: sortie pondérée par l'attention, de forme
            (batch, temps, taille_tete).
        """
        taille_lot, longueur_sequence, nb_canaux = x.shape
        k = self.cle(x)
        q = self.requete(x)
        poids_attention = q @ k.transpose(-2, -1) / (nb_canaux ** 0.5)
        poids_attention = poids_attention.masked_fill(
            self.triangle_inferieur[:longueur_sequence, :longueur_sequence] == 0, float('-inf')
        )
        poids_attention = F.softmax(poids_attention, dim=-1)
        v = self.valeur(x)
        sortie = poids_attention @ v
        return sortie


# ----------------------------
# Attention multi-têtes
# ----------------------------
class MultiHeadAttention(nn.Module):
    """
    Combine plusieurs têtes d'auto-attention exécutées en parallèle puis
    fusionne leurs sorties par une projection linéaire.
    """

    def __init__(self, dimension_embedding, taille_bloc, nb_tetes):
        """
        Args:
            dimension_embedding (int): dimension des vecteurs d'entrée/sortie.
            taille_bloc (int): longueur maximale de la séquence.
            nb_tetes (int): nombre de têtes d'attention en parallèle.
        """
        super().__init__()
        taille_tete = dimension_embedding // nb_tetes
        self.tetes = nn.ModuleList(
            [SelfAttentionHead(dimension_embedding, taille_bloc, taille_tete) for _ in range(nb_tetes)]
        )
        self.projection = nn.Linear(nb_tetes * taille_tete, dimension_embedding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tenseur d'entrée de forme (batch, temps, canaux).

        Returns:
            torch.Tensor: sortie combinée de toutes les têtes, reprojetée
            dans l'espace de dimension_embedding.
        """
        sortie = torch.cat([tete(x) for tete in self.tetes], dim=-1)
        return self.projection(sortie)


# ----------------------------
# Réseau feed-forward
# ----------------------------
class FeedForward(nn.Module):
    """
    Petit réseau entièrement connecté (MLP), appliqué indépendamment à
    chaque position, avec une couche cachée quatre fois plus large que
    l'embedding. Utilisé après l'attention dans chaque bloc Transformer.
    """

    def __init__(self, dimension_embedding):
        """
        Args:
            dimension_embedding (int): dimension d'entrée et de sortie du réseau.
        """
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(dimension_embedding, 4 * dimension_embedding),
            nn.ReLU(),
            nn.Linear(4 * dimension_embedding, dimension_embedding)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tenseur d'entrée.

        Returns:
            torch.Tensor: tenseur transformé par le MLP.
        """
        return self.reseau(x)


# ----------------------------
# Bloc Transformer
# ----------------------------
class Block(nn.Module):
    """
    Un bloc Transformer standard : attention multi-têtes puis réseau
    feed-forward, chacun précédé d'une normalisation (pre-LayerNorm) et
    suivi d'une connexion résiduelle.
    """

    def __init__(self, dimension_embedding, taille_bloc, nb_tetes):
        """
        Args:
            dimension_embedding (int): dimension des vecteurs traités par le bloc.
            taille_bloc (int): longueur maximale de la séquence (contexte).
            nb_tetes (int): nombre de têtes d'attention.
        """
        super().__init__()
        self.attention = MultiHeadAttention(dimension_embedding, taille_bloc, nb_tetes)
        self.reseau_ffwd = FeedForward(dimension_embedding)
        self.norme1 = nn.LayerNorm(dimension_embedding)
        self.norme2 = nn.LayerNorm(dimension_embedding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): tenseur d'entrée de forme (batch, temps, canaux).

        Returns:
            torch.Tensor: tenseur de même forme, après attention et
            feed-forward, avec connexions résiduelles.
        """
        x = x + self.attention(self.norme1(x))
        x = x + self.reseau_ffwd(self.norme2(x))
        return x
