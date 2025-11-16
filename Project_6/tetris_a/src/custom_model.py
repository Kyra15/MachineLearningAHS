from copy import copy, deepcopy
import numpy as np
import pygame
from piece import BODIES, Piece
from board import Board
import random
from genetic_helpers import *
from collections import deque


class CUSTOM_AI_MODEL:
    def __init__(self, genotype=None, aggregate='lin', num_features=11, mutate=False,  noise_sd=.2):

        if genotype is None:
            # ([peak_sum, holes, wells, bumpiness, height, barricades, cleared])
            self.genotype = np.array([random.uniform(-5,5) for _ in range(num_features)])
        else:
            if mutate == False:
                self.genotype = genotype.copy()
            else:
                # additive Gaussian mutation?
                self.genotype = genotype.copy()
                for i in range(len(self.genotype)):
                    if random.random() < 0.2:
                        self.genotype[i] += np.random.normal(0, 0.2)

        self.fit_score = 0.0
        self.fit_rel = 0.0
        self.aggregate = aggregate
        self.current_landing_height = 0

    
    def __lt__(self, other):
        return (self.fit_score<other.fit_score)

    
    def get_best_move(self, board, piece):
        """
        Gets the best for move an agents base on board, next piece, and genotype
        """

        best_x = -1000
        max_value = -1000
        best_piece = None
        rotated = piece
        for i in range(4):
            r = rotated
            
            for x in range(board.width):
                try:
                    y = board.drop_height(r, x)
                except:
                    continue

                board_copy = deepcopy(board.board)
                for pos in r.body:
                    board_copy[y + pos[1]][x + pos[0]] = True
                    
                self.current_landing_height = y
                np_board = bool_to_np(board_copy)
                c = self.valuate(np_board)

                if c > max_value:
                    max_value = c
                    best_x = x
                    best_piece = r
                    
            rotated = rotated.get_next_rotation()
            if best_piece is None:
                return 0, piece
            
        return best_x, best_piece


    def valuate(self, board):
        """
        """
        peaks = get_peaks(board)
        peak_sum = -np.sum(peaks)
        holes = -np.sum(get_holes(peaks, board))
        wells = -np.sum(get_wells(peaks))
        bumpiness = -get_bumpiness(peaks)
        height = -np.count_nonzero(np.mean(board, axis=1))
        barricades = -get_barricades(board)
        cleared = lines_cleared(board)
        col_transition = -get_col_transition(board, peaks)
        row_transition = -get_row_transition(board, max(peaks))
        max_peak = -max(peaks)
        landing_height = -self.current_landing_height
        

        ratings = np.array([
            peak_sum,
            holes,
            wells,
            bumpiness,
            height,
            barricades,
            cleared,
            col_transition,
            row_transition,
            max_peak,
            landing_height
        ])
        
        rating = np.dot(self.genotype, ratings)
        return rating


    
def get_peaks(area):
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)
    return peaks

    
def get_bumpiness(peaks):
    s = 0
    for i in range(9):
        s += np.abs(peaks[i] - peaks[i + 1])
    return s


def get_holes(peaks, area):
    # Count from peaks to bottom
    holes = []
    for col in range(area.shape[1]):
        start = -peaks[col]
        # If there's no holes i.e. no blocks on that column
        if start == 0:
            holes.append(0)
        else:
            holes.append(np.count_nonzero(area[int(start) :, col] == 0))
    return holes   


def get_barricades(area):
    h, w = area.shape
    visited = np.zeros_like(area, dtype=bool)
    
    q = deque()

    for col in range(w):
        if area[0, col] == 0:
            q.append((0, col))
            visited[0, col] = True

    # BFS flood fill
    while q:
        r, c = q.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if not visited[nr, nc] and area[nr, nc] == 0:
                    visited[nr, nc] = True
                    q.append((nr, nc))

    # cells not reachable from top
    return np.sum((area == 0) & (~visited))


def get_wells(peaks):
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0]
            w = w if w > 0 else 0
            wells.append(w)
        elif i == len(peaks) - 1:
            w = peaks[-2] - peaks[-1]
            w = w if w > 0 else 0
            wells.append(w)
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            w1 = w1 if w1 > 0 else 0
            w2 = w2 if w2 > 0 else 0
            w = w1 if w1 >= w2 else w2
            wells.append(w)
    return wells

def lines_cleared(area):
    cleared = 0
    for row in area:
        if np.all(row):
            cleared += 1
    return cleared


def get_row_transition(area, highest_peak):
    sum = 0
    # From highest peak to bottom
    for row in range(int(area.shape[0] - highest_peak), area.shape[0]):
        for col in range(1, area.shape[1]):
            if area[row, col] != area[row, col - 1]:
                sum += 1
    return sum


def get_col_transition(area, peaks):
    sum = 0
    for col in range(area.shape[1]):
        if peaks[col] <= 1:
            continue
        for row in range(int(area.shape[0] - peaks[col]), area.shape[0] - 1):
            if area[row, col] != area[row + 1, col]:
                sum += 1
    return sum


def min_max_norm(array, desired_min=0, desired_max=1):
    array_min = np.min(array)
    array_max = np.max(array)
    
    if array_max == array_min:
        return np.full_like(array, desired_min) 
    
    normalized_array = (array - array_min) / (array_max - array_min) * (desired_max - desired_min) + desired_min
    return normalized_array




        
            
