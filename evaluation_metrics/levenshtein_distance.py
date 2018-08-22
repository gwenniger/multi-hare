
"""
This class implements the Levenshtein distance based on pseudo code adapted from Wikipedia
https://en.wikipedia.org/wiki/Levenshtein_distance
And checked against a graphical implementation
http://www.let.rug.nl/kleiweg/lev/

It is assumed that the costs of deletion, insertion and substitution operations are all 1.

"""


def create_m_by_n_zeros_array(m: int, n: int):
    result = list([])
    for i in range(0, m):
        result.append([0] * n)
    return result


def print_matrix_array(matrix_array: list):
    for row_list in matrix_array:
        print(str(row_list))


def levenshtein_distance(symbol_list_one: list, symbol_list_two: list):
    """
    for all i and j, d[i,j] will hold the Levenshtein distance between
    the first i characters of s and the first j characters of t
    note that d has (m+1)*(n+1) values
    """
    m = len(symbol_list_one) + 1
    n = len(symbol_list_two) + 1
    d = create_m_by_n_zeros_array(m, n)

    """
    source prefixes can be transformed into empty string by
    dropping all characters
    """
    for i in range(1, m):
        d[i][0] = i

    """"  
    target prefixes can be reached from empty source prefix
    by inserting every character
    """

    for j in range(1, n):
        d[0][j] = j

    for j in range(1, n):
        for i in range(1, m):
            if symbol_list_one[i-1] == symbol_list_two[j-1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            d[i][j] = min(d[i-1][j] + 1,                    # deletion
                          d[i][j-1] + 1,                    # insertion
                          d[i-1][j-1] + substitution_cost)  # substitution

    print(str(symbol_list_one))
    print(str(symbol_list_two))
    print_matrix_array(d)

    result = d[m-1][n-1]
    print("Levenshtein distance: " + str(result))
    return result


def create_character_sequence_from_string(string: str):
    result = list([])
    result.extend(string)
    return result


def test_levenshtein_distance(char_sequence_one_as_string: str,
                              char_seq_two_as_string: str,
                              expected_distance: int):
    character_sequence_one = create_character_sequence_from_string(
        char_sequence_one_as_string)
    character_sequence_two = create_character_sequence_from_string(
        char_seq_two_as_string)
    distance = levenshtein_distance(character_sequence_one, character_sequence_two)

    if not distance == expected_distance:
        raise RuntimeError("Error: expected a levensthein distance of : "
                           + str(expected_distance) + " between: \"" +
                           char_sequence_one_as_string + "\" and \"" +
                           char_seq_two_as_string + "\" , but got: " + str(distance))


# See: http://www.let.rug.nl/kleiweg/lev/ for checking
def test_levenshtein_distance_example_one():
    # t h e 	  	  	m 	a 	n 	s 	a 	w 	t 	h 	e 	w 	o 	m a n w i t h t h e t e l o s c o p e
    # t h e 	w 	o 	m 	a 	n 	s 	a 	w 	t 	h 	e 	  	  	m a n w i t h t h e t e l o s c o p e
    #
    char_seq_one_as_string = "themansawthewomanwiththeteloscope"
    char_seq_two_as_string = "thewomansawthemanwiththeteloscope"
    test_levenshtein_distance(char_seq_one_as_string, char_seq_two_as_string, 4)

def test_levenshtein_distance_example_two():
    # a 	b 	c 	d 	e 	f 	g
    # a 	b 	  	d 	  	f 	h
    char_seq_one_as_string = "abcdefg"
    char_seq_two_as_string = "abdfh"
    test_levenshtein_distance(char_seq_one_as_string, char_seq_two_as_string, 4)


def main():
    test_levenshtein_distance_example_one()
    test_levenshtein_distance_example_two()


if __name__ == "__main__":
    main()
