import pygame, sys, os
import numpy as np
from scenes.MenuScene import MenuScene
from scenes.SceneManager import SceneManager
from scenes.MapEditorScene import MapEditor
from scenes.PlayGameScene import PlayGame
from scenes.TrainingScene import Training
from core.TextureAtlas import TextureAtlas
from core.CsvTilemap import CsvTileMap
from scenes.DuelScene import DuelScene
# Problematickej mixer resim pres SDL, ted vypnuto
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
## Start pygame + start modulů!
pygame.init()

######################################################################
# VS studio na windwos:
# tvorba virtual env: python3 -m venv .venv  
# aktivace linux: source .venv/bin/activate
# ctrl+shift+P - vybrat interpeter, ten nainstalovej - venv
# ve visual studio code F1 - pyhton select interpreter
# ############################################################
############################################################
############################################################
############################################################
### Students part:

#from AI_engines.AIbrain_TeamName import AIbrain_TeamName as trainbrain
#from AI_engines.AIbrain_linear import AIbrain_linear as trainbrain
from AI_engines.AIbrain_TymE import AIbrain_TymE as trainbrain


#### Students part end here.
############################################################
############################################################


# Konstatny
from constants import WIDTH, HEIGHT, FPS, tile_path_png, tile_path_xml, tile_path_default_setting_csv, TILESIZE
from constants import tilecicle, EMPTY_TILE, car_path_png, car_path_xml


# Nastaveni okna aj.
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI engines training")

# Grafika - preload dat
atlas = TextureAtlas(tile_path_png, tile_path_xml, scale_to=(TILESIZE, TILESIZE), scale_mode="smooth")
atlas.load()
atlas.convert_all()
tmap = CsvTileMap(
    atlas,
    tile_path_default_setting_csv,
    tile_w=TILESIZE,
    tile_h=TILESIZE,
    base_tile=EMPTY_TILE,   #  základní výplň mapy
    empty_symbol="."            #  POZOR! prázdné buňky v CSV znamenají „jen base“
)
tmap.prerender()

# vehicle:

vehicles_atlas = TextureAtlas(
    image_path=car_path_png,
    xml_path=car_path_xml,
    add_alias_without_ext=True,      # pak můžete psát jméno bez .png
    scale_to=(64, 128),              # nebo None pokud nechcete škálovat
    scale_mode="smooth",
)

vehicles_atlas.load()
vehicles_atlas.convert_all()


# hodiny - FPS CLOCK / heart rate
clock = pygame.time.Clock()

# Kolecke spritů
my_sprites = pygame.sprite.Group()

# start herní smyčky:
running = True

# Scene manager - rídí běh hry. to jaká "scéna" polozka menu je právě načtená
# scene manager a provtní setting  - registruji jak manager do scény tak scénu do managera
# SCENE manager je shell pro scény
scene_manager = SceneManager(screen)
scene_manager.set_cur_tmap(tmap) # pred pridáním scéne, jde do scén pak!
scene_manager.set_atlas_tmap(atlas)
scene_manager.set_default_tmap_name(tile_path_default_setting_csv)
scene_manager.set_TILESIZE(TILESIZE)
scene_manager.set_vehicle_atlas(vehicles_atlas)

# registrace polozek menu, vzdy  registruji do scene manager a manager naopak do scény
scene_manager.add_menu(MenuScene(scene_manager))
scene_manager.add_mapeditor(MapEditor(scene_manager, tilecicle))
scene_manager.add_playgame(PlayGame(scene_manager))
scene_manager.add_training(Training(scene_manager))
scene_manager.add_brain(trainbrain)

scene_manager.add_duel(DuelScene(scene_manager))



# cyklus udrzujici okno v chodu
while running:
    # FPS kontrola / jeslti bezi dle rychlosti!
    dt = clock.tick(FPS)/1000.0

    # Eventy a zavření okna:
    for event in pygame.event.get():
        # print(event) - pokud potrebujete info co se zmacklo.
        if event.type == pygame.QUIT:
            running = False
            break
        elif (not scene_manager.is_active()):
            running = False
            break
        else:
            # eventy prebrat jen kdyz orapvdu nezavírám, predtím musí být vzdy jinak break!!!
            scene_manager.event(event) # herní eventy handluje scene_manager který je predává dle potřeby.

    if not running:
        break

    # Update
    keys = pygame.key.get_pressed()
    scene_manager.update(dt, keys)



    # Render
    scene_manager.draw()
    pygame.display.flip()

print("Vypinani")
# POZOR! pokud chceme ukozit a pustit proces quitu ta NESMIME na linuxu zavolat event ci draw a nebo flip!!
# mením tím pak i šablonu do buduoucna - jank bude vyset process
# novinka, na linuxu obcas blokuje zavrení okna kdyz neukončím okn
# a další problém je kdyz vykrelsím do screnu ci udleám update proto dávám break
try:
    pygame.display.quit()
except Exception:
    pass

try:
    pygame.mixer.fadeout(200)
    pygame.time.delay(210)
    pygame.mixer.stop()
    pygame.mixer.quit()
except Exception:
    pass

pygame.quit()
sys.exit(0)
