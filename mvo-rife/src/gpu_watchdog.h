// Garde-fou d'exécution : perte du GPU et blocages.
//
// Après une perte du périphérique (VK_ERROR_DEVICE_LOST), le pilote peut laisser des fils bloqués indéfiniment dans des
// appels Vulkan (attente active) : le moteur ne rendait alors jamais la main. Le garde-fou, dans un fil dédié :
// - sonde chaque seconde une barrière jamais signalée de chaque périphérique utilisé (VK_ERROR_DEVICE_LOST après une perte) ;
// - détecte un blocage : aucun progrès (trame lue, restaurée, interpolée ou écrite) depuis `stall_seconds`, hors attente
//   de l'entrée ou de la sortie (amont ou aval lents ne sont pas des blocages) ;
// - après une erreur fatale (perte, blocage, échec signalé par fatal()), laisse `grace_seconds` au processus pour se
//   terminer normalement, puis le termine avec le code demandé (std::_Exit) : l'appelant voit un échec, jamais une attente
//   sans fin.
#pragma once
#include <cstdint>

namespace ncnn
{
class VulkanDevice;
}

namespace watchdog
{
// Démarre le fil (stall_seconds = 0 : pas de détection de blocage, sonde de perte conservée).
void start(int stall_seconds, int gpu_exit_code);
// Périphérique à sonder (plusieurs possibles ; doublons ignorés).
void watch_device(const ncnn::VulkanDevice* dev);
// Arrêt normal (fin du traitement).
void stop();
// Une unité de travail terminée (trame lue, restaurée, interpolée ou écrite).
void progress();
// Attente de l'entrée ou de la sortie en cours (compteur) : pas de détection de blocage pendant ce temps.
void io_begin();
void io_end();
// Erreur fatale : message (une fois), sortie forcée après le délai de grâce si le processus n'est pas terminé.
void fatal(int exit_code, const char* message);
// Vrai après une erreur fatale.
bool failed();
}  // namespace watchdog
