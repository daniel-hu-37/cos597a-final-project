from datetime import datetime

class Tester:
    def __init__(self):
        pass

    def test_graph_construction(self):
        print()
        print("Testing graph construction...")
        start = datetime.now()

        # Graph Construction here

        end = datetime.now()
        print("Time taken: ", end - start)
        print()
        pass

    def test_graph_search(self):
        pass

    def test_all(self):
        self.test_graph_construction()
        self.test_graph_search()

def main():
    tester = Tester()
    tester.test_all()

if __name__ == "__main__":
    main()