from copy import deepcopy
import time
import platform
# import io
from io import StringIO

# import resource
if platform.system() != 'Windows':
    import resource

import sys
import os

sys.path[0] = os.getcwd()


class Board:
    BLANK = " "
    BLOCKED = "X"
    NOT_MOVED = (-1, -1, False)

    # Flag True when swap occured last turn
    SWAP_FLAG = False

    __player_1__ = None
    __player_2__ = None
    __queen_1__ = None
    __queen_2__ = None

    __active_player__ = None
    __inactive_player__ = None
    __active_players_queen__ = None
    __inactive_players_queen__ = None

    __last_queen_move__ = {}
    __last_queen_symbols__ = {}

    move_count = 0

    def __init__(self, player_1, player_2, width=7, height=7):
        self.width = width
        self.height = height

        self.__player_1__ = player_1
        self.__player_2__ = player_2

        self.__queen_1__ = player_1.__class__.__name__ + " - Q1"
        self.__queen_2__ = player_2.__class__.__name__ + " - Q2"

        self.__board_state__ = [
            [Board.BLANK for i in range(0, width)] for j in range(0, height)]

        self.__last_queen_move__ = {
            self.__queen_1__: Board.NOT_MOVED, self.__queen_2__: Board.NOT_MOVED}

        self.__queen_symbols__ = {
            Board.BLANK: Board.BLANK, self.__queen_1__: "Q1", self.__queen_2__: "Q2"}

        self.__active_player__ = player_1
        self.__inactive_player__ = player_2
        self.__active_players_queen__ = self.__queen_1__
        self.__inactive_players_queen__ = self.__queen_2__

        self.move_count = 0

    def get_state(self):
        """
        Get physical board state
        Parameters:
            None
        Returns: 
            State of the board: list[char]
        """
        return deepcopy(self.__board_state__)

    def set_state(self, board_state, p1_turn=True):
        '''
        Function to immediately bring a board to a desired state. Useful for testing purposes; call board.play_isolation() afterwards to play
        Parameters:
            board_state: list[str], Desired state to set to board
            p1_turn: bool, Flag to determine which player is active
        Returns:
            None
        '''
        self.__board_state__ = board_state

        last_move_q1 = [(column,row.index("Q1"),0) for column, row in enumerate(board_state) if "Q1" in row]
        if (last_move_q1 != []):
            self.__last_queen_move__[self.__queen_1__] = last_move_q1[0] # set last move to the first found occurance of 'Q1'

        last_move_q2 = [(column,row.index("Q2"),0) for column, row in enumerate(board_state) if "Q2" in row]
        if (last_move_q2 != []):
            self.__last_queen_move__[self.__queen_2__] = last_move_q2[0]

        if (p1_turn):
            self.__active_player__ = self.__player_1__
            self.__active_players_queen__ = self.__queen_1__
            self.__inactive_player__ = self.__player_2__
            self.__inactive_players_queen__ = self.__queen_2__

        else:
            self.__active_player__ = self.__player_2__
            self.__active_players_queen__ = self.__queen_2__
            self.__inactive_player__ = self.__player_1__
            self.__inactive_players_queen__ = self.__queen_1__
        #Count X's to get move count + 2 for initial moves
        self.move_count = sum(row.count('X') + row.count('Q1') + row.count('Q2') for row in board_state)


    def __apply_move__(self, queen_move):
        '''
        Apply chosen move to a board state and check for game end
        Parameters:
            queen_move: (int, int, bool), Desired move to apply. Takes the 
            form of (row, column, does this move cause a swap or not).
        Returns:
            result: (bool, str), Game Over flag, winner 
        '''
        #print("Applying move:: ", queen_move)
        row, col, swap = queen_move
        my_pos = self.__last_queen_move__[self.__active_players_queen__]
        opponent_pos = self.__last_queen_move__[self.__inactive_players_queen__]

        queen_name = self.__queen_symbols__[self.__active_players_queen__]

        if (swap):
            # Apply swap move
            self.SWAP_FLAG = True
            opponent_new_pos = (my_pos[0], my_pos[1], True)
            self.__last_queen_move__[self.__inactive_players_queen__] = opponent_new_pos
            self.__board_state__[opponent_new_pos[0]][opponent_new_pos[1]] = \
                self.__queen_symbols__[self.__inactive_players_queen__]

        else:
            self.SWAP_FLAG = False
            if self.move_is_in_board(my_pos[0], my_pos[1]):
                self.__board_state__[my_pos[0]][my_pos[1]] = Board.BLOCKED

        # apply move of active player
        self.__last_queen_move__[self.__active_players_queen__] = queen_move
        self.__board_state__[row][col] = self.__queen_symbols__[self.__active_players_queen__]

        # If opponent is isolated
        if not self.get_inactive_moves():
            return True, self.__active_players_queen__

        # rotate the players
        self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__

        # rotate the queens
        self.__active_players_queen__, self.__inactive_players_queen__ = self.__inactive_players_queen__, self.__active_players_queen__

        # increment move count
        self.move_count = self.move_count + 1

        return False, None

    def copy(self):
        '''
        Create a copy of this board and game state.
        Parameters:
            None
        Returns:
            Copy of self: Board class
        '''
        b = Board(self.__player_1__, self.__player_2__,
                  width=self.width, height=self.height)
        for key, value in self.__last_queen_move__.items():
            b.__last_queen_move__[key] = value
        for key, value in self.__queen_symbols__.items():
            b.__queen_symbols__[key] = value
        b.move_count = self.move_count
        b.__active_player__ = self.__active_player__
        b.__inactive_player__ = self.__inactive_player__
        b.__active_players_queen__ = self.__active_players_queen__
        b.__inactive_players_queen__ = self.__inactive_players_queen__
        b.__board_state__ = self.get_state()
        b.SWAP_FLAG = self.SWAP_FLAG
        return b

    def forecast_move(self, queen_move):
        """
        See what board state would result from making a particular move without changing the board state itself.
        Parameters:
            queen_move: (int, int, bool), Desired move to forecast. Takes the form of
            (row, column, does this move cause a swap or not).

        Returns:
            (Board, bool, str): Resultant board from move, flag for game-over, winner (if game is over)
        """
        new_board = self.copy()
        is_over, winner = new_board.__apply_move__(queen_move)
        return new_board, is_over, winner

    def get_active_player(self):
        """
        See which player is active. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            str: Name of the player who's actively taking a turn
        """
        return self.__active_player__

    def get_inactive_player(self):
        """
        See which player is inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            str: Name of the player who's waiting for opponent to take a turn
        """
        return self.__inactive_player__

    def get_active_players_queen(self):
        """
        See which queen is inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            str: Queen name of the player who's waiting for opponent to take a turn
        """
        return self.__active_players_queen__

    def get_inactive_players_queen(self):
        """
        See which queen is inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            str: Queen name of the player who's waiting for opponent to take a turn
        """
        return self.__inactive_players_queen__

    def get_inactive_position(self):
        """
        Get position of inactive player (player waiting for opponent to make move) in [row, column] format
        Parameters:
            None
        Returns:
           [int, int]: [row,col] of inactive player
        """
        return self.__last_queen_move__[
            self.__inactive_players_queen__][0:2]

    def get_active_position(self):
        """
        Get position of active player (player actively making move) in [row, column] format
        Parameters:
            None
        Returns:
           [int, int]: [row,col] of inactive player
        """
        return self.__last_queen_move__[
            self.__active_players_queen__][0:2]

    def get_player_position(self, my_player=None):
        """
        Get position of certain player object. Should pass in yourself to get your position.
        Parameters:
            my_player (Player), Player to get position for
            If calling from within a player class, my_player = self can be passed.
        returns
            [int, int]: [Row, Col] position of player

        """
        if (my_player == self.__player_1__ and self.__active_player__ == self.__player_1__):
            return self.get_active_position()
        if (my_player == self.__player_1__ and self.__active_player__ != self.__player_1__):
            return self.get_inactive_position()
        elif (my_player == self.__player_2__ and self.__active_player__ == self.__player_2__):
            return self.get_active_position()
        elif (my_player == self.__player_2__ and self.__active_player__ != self.__player_2__):
            return self.get_inactive_position()
        else:
            raise ValueError("No value for my_player!")


    def get_opponent_position(self, my_player=None):
        """
        Get position of my_player's opponent.
        Parameters:
            my_player (Player), Player to get opponent's position
            If calling from within a player class, my_player = self can be passed.
        returns
            [int, int]: [Row, col] position of my_player's opponent

        """
        if (my_player == self.__player_1__ and self.__active_player__ == self.__player_1__):
            return self.get_inactive_position()
        if (my_player == self.__player_1__ and self.__active_player__ != self.__player_1__):
            return self.get_active_position()
        elif (my_player == self.__player_2__ and self.__active_player__ == self.__player_2__):
            return self.get_inactive_position()
        elif (my_player == self.__player_2__ and self.__active_player__ != self.__player_2__):
            return self.get_active_position()
        else:
            raise ValueError("No value for my_player!")

    def get_inactive_moves(self):
        """
        Get all legal moves of inactive player on current board state as a list of possible moves.
        Parameters:
            None
        Returns:
           [(int, int, bool)]: List of all legal moves. Each move takes the form of
            (row, column, does this move cause a swap or not).
        """
        q_move = self.__last_queen_move__[
            self.__inactive_players_queen__]

        return self.__get_moves__(q_move)

    def get_active_moves(self):
        """
        Get all legal moves of active player on current board state as a list of possible moves.
        Parameters:
            None
        Returns:
           [(int, int, bool)]: List of all legal moves. Each move takes the form of
            (row, column, does this move cause a swap or not).
        """
        q_move = self.__last_queen_move__[
            self.__active_players_queen__]

        return self.__get_moves__(q_move)

    def get_player_moves(self, my_player=None):
        """
        Get all legal moves of certain player object. Should pass in yourself to get your moves.
        Parameters:
            my_player (Player), Player to get moves for
            If calling from within a player class, my_player = self can be passed.
        returns
            [(int, int, bool)]: List of all legal moves. Each move takes the form of
            (row, column, does this move cause a swap or not).

        """
        if (my_player == self.__player_1__ and self.__active_player__ == self.__player_1__):
            return self.get_active_moves()
        if (my_player == self.__player_1__ and self.__active_player__ != self.__player_1__):
            return self.get_inactive_moves()
        elif (my_player == self.__player_2__ and self.__active_player__ == self.__player_2__):
            return self.get_active_moves()
        elif (my_player == self.__player_2__ and self.__active_player__ != self.__player_2__):
            return self.get_inactive_moves()
        else:
            raise ValueError("No value for my_player!")

    def get_opponent_moves(self, my_player=None):
        """
        Get all legal moves of the opponent of the player provided. Should pass in yourself to get your opponent's moves.
        If calling from within a player class, my_player = self can be passed.
        Parameters:
            my_player (Player), The player facing the opponent in question
            If calling from within a player class, my_player = self can be passed.
        returns
            [(int, int, bool)]: List of all opponent's moves. Each move takes the form of
            (row, column, does this move cause a swap or not).

        """
        if (my_player == self.__player_1__ and self.__active_player__ == self.__player_1__):
            return self.get_inactive_moves()
        if (my_player == self.__player_1__ and self.__active_player__ != self.__player_1__):
            return self.get_active_moves()
        elif (my_player == self.__player_2__ and self.__active_player__ == self.__player_2__):
            return self.get_inactive_moves()
        elif (my_player == self.__player_2__ and self.__active_player__ != self.__player_2__):
            return self.get_active_moves()
        else:
            raise ValueError("No value for my_player!")

    def __get_moves__(self, move):
        """
        Get all legal moves of a player on current board state as a list of possible moves. Not meant to be directly called, 
        use get_active_moves or get_inactive_moves instead.
        Parameters:
            move: (int, int, bool), Last move made by player in question (where they currently are). 
            Takes the form of (row, column, does this move cause a swap or not).
        Returns:
           [(int, int, bool)]: List of all legal moves. Each move takes the form of
            (row, column, does this move cause a swap or not).
        """

        if move == self.NOT_MOVED:
            return self.get_first_moves()

        r, c, _ = move

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1),          ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)]

        moves = []


        for direction in directions:
            for dist in range(1, max(self.height, self.width)):
                row = direction[0] * dist + r
                col = direction[1] * dist + c
                if self.move_is_in_board(row, col) and self.is_spot_open(row, col) and (row, col, False) not in moves:
                    moves.append((row, col, False))

                elif self.move_is_in_board(row, col) and self.is_spot_queen(row, col):
                    for distance in range(1, dist + 1):
                        if self.does_move_allow_swap(row, col, direction, distance) and not self.SWAP_FLAG and(row, col, True) not in moves:
                            moves.append((row, col, True))
                        else:
                            break
                    break
                else:
                    break

        return moves

    def get_first_moves(self):
        """
        Return all moves for first turn in game (i.e. every board position)
        Parameters:
            None
        Returns:
           [(int, int, bool)]: List of all legal moves. Each move takes the form of
            (row, column, does this move cause a swap or not).
        """
        return [(i, j, 0) for i in range(0, self.height)
                          for j in range(0, self.width) if self.__board_state__[i][j] == Board.BLANK]

    def move_is_in_board(self, row, col):
        """
        Sanity check for making sure a move is within the bounds of the board.
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
        Returns:
            bool: Whether the [row,col] values are within valid ranges
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def is_spot_open(self, row, col):
        """
        Sanity check for making sure a move isn't occupied by an X.
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
        Returns:
            bool: Whether the [row,col] position is blank (no X)
        """
        return self.__board_state__[row][col] == Board.BLANK

    def is_spot_queen(self, row, col):
        """
        Sanity check for checking if a spot is occupied by a player
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
        Returns:
            bool: Whether the [row,col] position is currently occupied by a player's queen
        """
        q1 = self.__queen_symbols__[self.__active_players_queen__]
        q2 = self.__queen_symbols__[self.__inactive_players_queen__]
        return self.__board_state__[row][col] == q1 or self.__board_state__[row][col] == q2

    def does_move_allow_swap(self, row, col, direction, distance):

        """
        Sanity check for checking if a move is a valid swap move
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
            direction: (int, int), Directional unit vector of incoming queen
            distance: int, Distance between [row,col] and incoming queen
        Returns:
            bool: Whether the [row,col] move is a valid swap move
        """

        if not self.is_spot_queen(row, col):
            return False

        # All spots up to the other queen to must be available to move to and be on the board
        for dist in range(1, distance - 1):
            row_target = row + direction[0] * dist
            col_target = col + direction[1] * dist
            if not self.move_is_in_board(row_target, col_target) or not self.is_spot_open(row_target, col_target):
                return False

        '''
        row_target = row + direction[0] * distance
        col_target = col + direction[1] * distance

        if (row_target > self.height) or (row_target < -1) or \
                (col_target > self.width) or (col_target < -1):
            return False

        if not self.move_is_in_board(row_target, col_target) or self.is_spot_open(row_target, col_target):
            # and IF the square they'd be pushed TO is either off the board or an open space on the board
            return True
        '''

        return True

    def space_is_open(self, row, col):
        """
        Sanity check to see if a space is within the bounds of the board and blank. Not meant to be called directly if you don't know what 
        you're looking for.
        Parameters:
            row: int, Row value of desired space
            col: int, Col value of desired space
        Returns:
            bool: (Row, Col ranges are valid) AND (space is blank)
        """
        return 0 <= row < self.height and \
               0 <= col < self.width and \
               self.__board_state__[row][col] == Board.BLANK

    def print_board(self, legal_moves=[]):
        """
        Function for printing board state & indicating possible moves for active player.
        Parameters:
            legal_moves: [(int, int, bool)], List of legal moves to indicate when printing board spaces. 
            Each move takes the form of (row, column, does this move cause a swap or not).
        Returns:
            Str: Visual interpretation of board state & possible moves for active player
        """

        p1_r, p1_c, swap = self.__last_queen_move__[self.__queen_1__]
        p2_r, p2_c, swap = self.__last_queen_move__[self.__queen_2__]
        b = self.__board_state__

        out = '  |'
        for i in range(len(b[0])):
            out += str(i) + ' |'
        out += '\n\r'

        for i in range(len(b)):
            out += str(i) + ' |'
            for j in range(len(b[i])):
                if (i, j) == (p1_r, p1_c):
                    out += self.__queen_symbols__[self.__queen_1__]
                elif (i, j) == (p2_r, p2_c):
                    out += self.__queen_symbols__[self.__queen_2__]
                elif (i, j, True) in legal_moves or (i, j, False) in legal_moves:
                    out += 'o '
                elif b[i][j] == Board.BLANK:
                    out += '  '
                else:
                    out += '><'

                out += '|'
            if i != len(b) - 1:
                out += '\n\r'

        return out

    def play_isolation(self, time_limit=10000, print_moves=False):
        """
        Method to play out a game of isolation with the agents passed into the Board class.
        Initializes and updates move_history variable, enforces timeouts, and prints the game.
        Parameters:
            time_limit: int, time limit in milliseconds that each player has before they time out.
            print_moves: bool, Should the method print details of the game in real time
        Returns:
            (str, [(int, int, bool)], str): Queen of Winner, Move history, Reason for game over.
            Each move in move history takes the form of (row, column, does this move cause a swap or not).
        """
        move_history = []

        if platform.system() == 'Windows':
            def curr_time_millis():
                return int(round(time.time() * 1000))
        else:
            def curr_time_millis():
                return 1000 * resource.getrusage(resource.RUSAGE_SELF).ru_utime

        while True:
            game_copy = self.copy()
            move_start = curr_time_millis()

            def time_left():
                # print("Limit: "+str(time_limit) +" - "+str(curr_time_millis()-move_start))
                return time_limit - (curr_time_millis() - move_start)

            if print_moves:
                print("\n", self.__active_players_queen__, " Turn")

            if (self.SWAP_FLAG):
                game_copy.SWAP_FLAG = True
            else:
                game_copy.SWAP_FLAG = False
            
            curr_move = self.__active_player__.move(
                game_copy, time_left)  # queen added in return

            # Append new move to game history
            if self.__active_player__ == self.__player_1__:
                move_history.append([curr_move])
            else:
                move_history[-1].append(curr_move)

            # Handle Timeout
            if time_limit and time_left() <= 0:
                return self.__inactive_players_queen__, move_history, \
                       (self.__active_players_queen__ + " timed out.")

            # Safety Check
            legal_moves = self.get_active_moves()
            if curr_move not in legal_moves:
                return self.__inactive_players_queen__, move_history, \
                       (self.__active_players_queen__ + " made an illegal move.")

            # Apply move to game.
            is_over, winner = self.__apply_move__(curr_move)

            if print_moves:
                print("move chosen: ", curr_move)
                print(self.copy().print_board())

            if is_over:
                if not self.get_inactive_moves():
                    return self.__active_players_queen__, move_history, \
                           (self.__inactive_players_queen__ + " has no legal moves left.")
                return self.__active_players_queen__, move_history, \
                       (self.__inactive_players_queen__ + " was forced off the grid.")

    def __apply_move_write__(self, move_queen):
        """
        Equivalent to __apply_move__, meant specifically for applying move history to a board 
        for analyzing an already played game.
        Parameters: 
            move_queen: (int, int, bool), Move to apply to board. Takes
            the form of (row, column, does this move cause a swap or not).
        Returns:
            None
        """

        if move_queen[0] is None or move_queen[1] is None:
            return

        row, col, swap_flag = move_queen
        my_pos = self.__last_queen_move__[self.__active_players_queen__]
        opponent_pos = self.__last_queen_move__[self.__inactive_players_queen__]

        if (move_queen[2]):
            # Apply swap move
            opponent_new_pos = (my_pos[0], my_pos[1], True)
            self.__last_queen_move__[self.__inactive_players_queen__] = opponent_new_pos
            self.__last_queen_move__[self.__active_players_queen__] = (row, col, True)
            self.__board_state__[opponent_new_pos[0]][opponent_new_pos[1]] = \
                self.__queen_symbols__[self.__inactive_players_queen__]
            self.__board_state__[row][col] = \
                self.__queen_symbols__[self.__active_players_queen__]
        else:
            self.__last_queen_move__[self.__active_players_queen__] = move_queen
            self.__board_state__[row][col] = \
                self.__queen_symbols__[self.__active_players_queen__]

        if self.move_is_in_board(my_pos[0], my_pos[1]):
            self.__board_state__[my_pos[0]][my_pos[1]] = Board.BLOCKED


        # swap the players
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp

        # swapping the queens
        tmp = self.__active_players_queen__
        self.__active_players_queen__ = self.__inactive_players_queen__
        self.__inactive_players_queen__ = tmp

        self.move_count = self.move_count + 1


def game_as_text(winner, move_history, termination="", board=Board(1, 2)):
    """
    Function to play out a move history on a new board. Used for analyzing an interesting move history 
    Parameters: 
        move_history: [(int, int, bool)], History of all moves in order of game in question. 
        Each move takes the form of (row, column, does this move cause a swap or not).
        termination: str, Reason for game over of game in question. Obtained from play_isolation
        board: Board, board that game in question was played on. Used to initialize board copy
    Returns:
        Str: Print output of move_history being played out.
    """
    ans = StringIO()

    board = Board(board.__player_1__, board.__player_2__, board.width, board.height)

    print("Printing the game as text.")

    last_move = (9, 9, 0)

    for i, move in enumerate(move_history):
        if move is None or len(move) == 0:
            continue

        if move[0] != Board.NOT_MOVED and move[0] is not None:
            ans.write(board.print_board())
            board.__apply_move_write__(move[0])
            ans.write("\n\n" + board.__queen_1__ + " moves to (" + str(move[0][0]) + "," + str(move[0][1]) + ")\r\n")

            if len(move) > 1 and move[0][2] is True:
                my_x, my_y = last_move[0][0], last_move[0][1]
                enemy_x, enemy_y = move[0][0], move[0][1]

                new_enemy_x, new_enemy_y = my_x, my_y

                ans.write(
                    "\n\n" + board.__queen_2__ + " swapped to (" + str(new_enemy_x) + "," + str(new_enemy_y) + ")\r\n")

        if len(move) > 1 and move[1] != Board.NOT_MOVED and move[0] is not None:
            ans.write(board.print_board())
            board.__apply_move_write__(move[1])
            ans.write("\n\n" + board.__queen_2__ + " moves to (" + str(move[1][0]) + "," + str(move[1][1]) + ")\r\n")

            if move[1] is not None and move[1][2] is True:
                my_x, my_y = last_move[1][0], last_move[1][1]
                new_enemy_x, new_enemy_y = move[1][0], move[1][1]
                new_enemy_x, new_enemy_y = my_x, my_y

                ans.write(
                    "\n\n" + board.__queen_1__ + " swapped to (" + str(new_enemy_x) + "," + str(new_enemy_y) + ")\r\n")

        last_move = move

    ans.write("\n" + str(winner) + " has won. Reason: " + str(termination))
    return ans.getvalue()
