import xml.etree.ElementTree as ET
import glob

class xml_obj_det():
    def __init__(self,main_dir_path):
        """
        This class uses previously generated xml files in annotating images in object detections and prepare labels dynamically,
         and path to image in order to be used by your ML package later!
         All you have to do is use locate your files and get your images path using get_files_paths() for X, and get_combined_labels() for Y!!
         After fitting the training set, use fit_data() on your test set as well to prepare your test set, "Don't worry your labels will be the same".
        :param main_dir_path:
            The Path where your train files are located in your machine, "images,xml"
        """
        self.__object = None
        self.__bbox = None
        self.path = main_dir_path
        self.__file = None
        self.__tree = None
        self.__root = None
        self.__files_path = []
        self.__bndboxes = []
        self.__diminsions = []
        self.__labels = []
        self.__files_names = []
        self.__uniq_labels = []
        self.__combined_labels=[]
        self.__num_labels=0
        # To handle the test set based on the train set
        self.__num_fit=0
        self.fit_data()
    # ----------------------------------
    def fit_data(self, path=None):
        """
        This methode fit the class to the images and prepare dynamically generated labels and path to where the xml and images files are
        :param path:
            Is the location where is your images/xml files located, if not used it will use the inital path used in the constructor methode
        :return:
            Return nothing
        """
        if path == None:
            path = self.path
        else:
            path = path
        if self.__num_fit > 0:
            self.__rest_data()
        self.__num_fit+=1
        for file_path in glob.glob(f'{path}/*.xml'):
            self.__add_file(file_path)
            file_name = self.__get_file_name()
            self.__files_names.append(file_name)
            self.__files_path.append(f"{path}/{file_name}")
            bndboxe = self.__get_bndboxe()
            self.__bndboxes.append(bndboxe)
            diminsions = self.__get_height_width()
            self.__diminsions.append(diminsions)
            label = self.__get_label()
            self.__labels.append(label)
            self.__set_uniq_labels(label)
        self.__encode_labels()
        self.__combine_labels()
    # ---------------------------------
    def __rest_data(self):
        self.__files_path = []
        self.__bndboxes = []
        self.__diminsions = []
        self.__labels = []
        self.__files_names = []
        self.__combined_labels = []
    # -----------------------------
    def __add_file(self,file_path):
        """
        This function adds file
        """
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.object = self.root.find("object")
    # -------------------------------------
    def __get_height_width(self):
        sizes = self.root.find("size")
        width = float(sizes[0].text)
        height = float(sizes[1].text)
        depth = float(sizes[2].text)
        return [width,height,depth]
    # ----------------------------------
    def __get_bndboxe(self):
        bndbox = self.object.find("bndbox")
        xmin = float(bndbox[0].text)
        ymin = float(bndbox[1].text)
        xmax = float(bndbox[2].text)
        ymax = float(bndbox[3].text)

        return [xmin,ymin,xmax,ymax]
    # ----------------------------------
    def __get_file_name(self):
        return self.root.find("filename").text
    # ---------------------------------
    def __set_uniq_labels(self,label):
        if not label in self.__uniq_labels:
            self.__uniq_labels.append(label)
            self.__num_labels+=1
    # ----------------------------------
    def __get_label(self):
        object = self.object
        label = object[0].text
        return label
    # ---------------------------
    def __encode_labels(self):
        for i,label in enumerate(self.__labels):
            self.__labels[i] = self.__uniq_labels.index(label)
    # ---------------------------
    def __combine_labels(self):
        for i in range(len(self.__labels)):
            bndboxe = self.__bndboxes[i]
            bndboxe.extend([self.__labels[i]])
            combine_label = bndboxe
            self.__combined_labels.append(combine_label)
    # -----------------------
    def get_bndboxes(self):
        return self.__bndboxes
    # ---------------------
    def get_diminsions(self):
        return self.__diminsions
    # ---------------------
    def get_labels(self):
        return self.__labels
    # ---------------------
    def get_files_paths(self):
        return self.__files_path
    # ----------------------
    def get_files_names(self):
        return self.__files_names
    def get_num_labels(self):
        return self.__num_labels
    # ------------------------
    def get_combined_labels(self):
        return self.__combined_labels

