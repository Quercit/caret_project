library(shiny)
library(shinydashboard)
library(tidyverse)

################# TEXT #########################
machine_learning_1 <-  "Machine learning is a name for algorithms that use computational methods to acquire information directly from the data without relying on a predetermined equation a as a model; that is why we refer to it as learning. These algorithms adaptively improve their performance as the number of samples available for learning increases. Simply put, the machine learning algorithms will find natural patterns within the data, get insights about the patterns, and make the predictions about the nature of new data when tested. Based on how much human input they need for learning the patterns of the data, we can make a distinction between two machine learning techniques: supervised learning and unsupervised learning."
machine_learning_2 <- "Supervised learning refers to finding patterns in the input data, and predicting the patterns of the output data. The algorithms from this family of techniques always need to be provided with a labeled dataset from which they can learn the interactions between the parameters and labels. For example, if we want to predict whether it will rain in Glasgow on Saturday, we will be using machine learning to solve a classification problem. For that, we will need a dataset which contains information on whether it did or didn't rain on a certain day in the last 10 years. The machine will then connect these labels, 'raining' and 'non-raining', to the day of the year, and based on that predict whether we will have rain on Saturday. In this case, we are making a binary classification, because we are making a prediction between two classes. If we would make a dataset with the labels 'raining', 'cloudy', 'sunny', and 'snowing', then we would be making a multi-class prediction. Alternatively, we could be interested in predicting the temperature in Glasgow on Sunday, based on the information about the temperature on the given day during the last ten years. This would be a problem of regression, because we would be predicting a continuous outcome.  "
machine_learning_3 <- "This simplified example of weather forecasting based on one predictor illustrates the basic logic of machine learning, but definitely doesn't show its full potential. Machine learning becomes especially useful when we are dealing with high-dimensional datasets, sometimes containing thousands and thousands of predictors, because we don't need to explicitly define the relationship between the predictors; the machine will learn the patterns directly from the data. Oftentimes the machine won't even need the labeled dataset; it will search through a new, unlabeled dataset, and based on patterns of interactions between datasets define distinct clusters. For example, if we collect the data about the temperature in Glasgow on random days throughout 10 years, but don't have an information about the day on which the temperature was measured, machine learning will help us separate this dataset into four distinct categories with varying temperature ranges: spring, summer (if such a thing exists in Scotland), autumn, and winter. This is referred to as cluster analysis, and is only one type of unsupervised learning."

split_data <- "The goal of machine learning is to train the data to predict future data. To do so, we want to first validate the model and check how accurately it can give us information about the new data. However, the information about the new data is not yet known; if it was known, we wouldn???t need the machine in the first place! This is why we usually split the existing, labeled dataset during the process of training and validating the model. This allows us to get an estimate of the accuracy with which we will be predicting all the new, unknown, unlabeled data. The usual approach is to split the dataset in two parts, hence forming a training set and test set. One of the common ways are to apply the Pareto principle and use 80% of the dataset for training, and 20% for testing. Alternatively, ratios of 66/33 or 50/50 are also sometimes used. The interactive graph shows how the [ WHAT CHANGES? ] changes when we split our example dataset in different ratios."



knn_dummies <- "The K-nearest neighbour algorithm (kNN) can be used both for classification and regression. In the case of classification, we can use a labeled data set to train the model to predict the classes of the data points in the test set. Interestingly, no explicit training step is required because the algorithm uses the whole training set into account when predicting the class membership of the test data point; this is referred to as lazy learning. 
                The algorithm makes the prediction by searching through the entire training set for the k number of similar instances (so-called neighbours), and based on their class membership predicts the class of the new data point.
                The value for k can be found by algorithm tuning, but it is recommended to try many different values for k and see what works best for your problem. We also use continuous weights called kernels to reflect the distance of a training point and the corresponding prediction point. Instead of using simple binary (i.e., neighbour/not-neighbour) distinctions, continuous kernels can capture the ???degree of neighbourship???). Basically, the closer two data points are, the greater the kernel value. 
                You can check the effect of different values of k on accuracy for different kernels in the graph. "

knn_smarties <- "The k- nearest neighbours (kNN) is a non-parametric method used for both classification and regression problem. The input consists of the k closest training examples in the feature space, and the output is either class membership or property value for the object, depending on problem type. In classification, the object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors. k is a positive integer, and typically small; for example, when k = 1, the object is simply assigned to the class of that single nearest neighbor.
                Another parameter that can be manipulated when setting up the model is the type and width of the kernel, which reflects the distance between the training data point and the corresponding test point. The commonly used kernels are Gaussian, regularized, and [third name] kernel. 
                An important feature of kernels is the kernel width (??). If the kernel width is small, the model will average over fewer points, and the bias will be smaller since closer distances are used; however, in this scenario we risk of the predictions being too bumpy. On the other side, if the kernel width is large, the model will be averaging over larger points and we will you have a larger bias since further apart distances are used; hence, we risk over-smoothing the parameter space. Therefore, it is important to consider the ???right??? trade-off depending on your model. 
                For determining the optimal k instances, the most commonly used method is the Euclidian distance, the square root of the sum of the squared differences between a new point (x) and an existing point (xi).
                EuclideanDistance(x, xi) = sqrt( sum( (xj ??? xij)^2 ) )
                The graph shows the effect of different k values and three most commonly used kernels on model accuracy."

svm_dummies <- "SVMs are supervised learning models that can be used both for regression and classification. In a training set, where all datapoints can clearly be distinguished between two groups, a SVM builds a model on this data (training set) that can be used for new data, to predict the assignment to one category or another. Two parallel hyperplanes can be selected that divide the two groups of each other. The goal is to maximize the distance between them. The distance of the both hyperplanes is the margin. The maximum-margin hyperplane is the hyperplane that lies halfway between them.
                In a dataset,  (x1???,y2), ???., (x???,yn), we want to find the maximum-margin hyperplane that divides the group of points (with groups that are either yi = 1 or yi = -1) into a maximised distance between the hyperplane and the nearest point xi??? from either group. Every hyperplane can be written down as w???*x???-b=0, where w??? describes the normal vector to the hyperplane. Therefore, the equation can only have the outcomes w???*x???-b=1 and w???*x???-b=-1 that determine the class above or below the decision boundary."

svm_smarties <- "Given  a dataset of 1000 photographs of pandas and sloths, you can digitize them into 500x500 pixels. So you have x ??? R n with n=250.000. Given another picture, you want to assess whether it is a panda or a sloth (Assumed supervised learning between two outcomes).
                Given your input/output sets (X,Y) and the training set (x1,y1)...,(xi,yi) and your previous  x ??? X you want to generalize to a suitable y ??? Y
                You then train a classifier with y= f(x,??), where ?? are your parameters of interest of the function.
                A proper training function for f(x,??) is crucial. So you choose a function that suits well. Eg. by  choosing: Rempirical_risk(??)= (1/m)*L*(f(xi,??),yi) for your Training error where L is your zero-one loss function, you can try to minimize the risk by setting
                R(??)=??? L(f(x,??),y)dP(x,y) for your Test error where P(x,y) is the distribution of x and y."

rf_dummies <- "Random forest is one of the most popular classification algorithms and can be used for both classification and regression. It is based on a concept of a decision tree - and that???s what makes the algorithm name very witty. In a decision tree, we go through a flowchart of questions, and depending on the answer to each question, we keep adding up new ones. In that way, we predict the outcome based on the pathway chosen through the nodes of the tree. For example, imagine you want to predict if you are going to pass your next exam. You may start asking yourself if you???ve spent enough hours studying. If that number of is less than, say, 20 hours, you might ask how much you knew beforehand, and follow-up by questioning the teaching quality, or how difficult was the exam in the previous years. After several questions, you will have a class prediction for your exam: pass or fail. By adding additional trees, you can see how many of them predict a fail and how many predict a pass. Since these are smart, machine trees, they will take into account a number of random features and their interactions. The more trees you put in your forest, the more opinions you will have, and the more accurate your final prediction will be. You can check our graph to see how the number of trees relates to the accuracy of the model prediction."
rf_smarties <- "For every kth tree, a random and independent vector k is generated. Every tree grows by using the training set with the help of the classifier h(x,k) where x is the input vector. After enough trees have been generated, the collection can be called a random forest. The random forest is still  a classifier that consist of a subcollection off tree-classifiers {h(x,k),k=1,...,n} where every tree votes for its self-generated prediction.
                In an ensemble of tree classifiers h1(x),...,hk(x) and a random training set from the vector distribution Y,X, the margin measures to what extent the mean number of votes of X,Y for the correct class exceeds the mean of any other class. Therefore: The larger the margin, the better the classification. In mathematical terms the margin can defined as following:
                margin(X,Y)=meank*I(hk(X)=Y)-max(j/=Y)*meank*I(hk(X)=j)
                Remarkable is that random forest do not overfit the data when additional trees are added. However, a limit of the predicted generalization error will be reached."

# header --------------------
header <- dashboardHeader(title = "Machine Learning Algorithms")


# sidebar -------------------
sidebar <- dashboardSidebar(
  sidebarMenu(
    menuItem("Introduction", tabName = "intro"),
    menuItem("Our Dataset", tabName = "dataset"),
    menuItem("Our Algorithms", tabName = "algorithms"),
    menuItem("Compare the models!", tabName = "compare"),
    menuItem("Gamify", tabName = "game")
  )
)


# body ----------------
body <- dashboardBody(
  tabItems(
    
    tabItem(tabName = "intro",
            tabsetPanel(
              tabPanel("Basic", 
                       box(title = "What is machine learning",
                           width = 6,
                           status = "success",
                           solidHeader = T,
                           machine_learning_1)
              ),
              tabPanel("Supervised", 
                       box(title = "Supervised Learning",
                           width = 6,
                           status = "success",
                           solidHeader = T,
                           machine_learning_2
                       )
              ),
              tabPanel("Unsupervised", 
                       box(title = "Unsupervised Learning",
                           width = 6,
                           status = "success",
                           solidHeader = T,
                           machine_learning_3
                       )
              ),
              tabPanel("Split the data", 
                       box(title = "Split your dataset",
                           width = 6,
                           status = "success",
                           solidHeader = T
                           
                       )
              ),
              tabPanel("", 
                       box(title = "gg",
                           width = 6,
                           status = "success",
                           solidHeader = T,
                           machine_learning_3
                       )
              )
            )
    ),
    
    tabItem(tabName = "dataset"),
    
    tabItem(tabName = "algorithms",
              tabsetPanel(
                tabPanel("knn",
                         tabBox(
                           width = 4,
                           selected = "Plain English",
                           tabPanel("Plain English", knn_dummies),
                           tabPanel("Jargon", knn_smarties),
                           tabPanel("Example",
                             "Test data details:",
                             tableOutput("testData"),
                             "Test data quality prediction:",
                             verbatimTextOutput("testQuality"),
                             "Test data real quality:",
                             verbatimTextOutput("realQuality"),
                             actionButton("randomize", "Randomize data!")
                          )
                         ),
                         box(
                           title = "Plot",
                           width = 8,
                           status = "success", 
                           solidHeader = TRUE,
                           radioButtons("vsize", "Select the training sample size:", 
                                        c("Pareto (80/20)" = "eightyTwenty",
                                          "66/33" = "oneThird",
                                          "50/50" = "halfHalf"),
                                        inline = TRUE
                           ),
                           checkboxGroupInput("vkernel", "Select kernel(s):",
                                              choiceNames = list("Rectangular", "Gaussian", "Cosine"),
                                              choiceValues = list("rectangular", "gaussian", "cos"),
                                              selected = "rectangular", inline = TRUE),
                           plotOutput("kknnplot1")
                         )
                ),
                tabPanel("MLP",
                         tabBox(
                           width = 4,
                           selected = "Plain English",
                           tabPanel("Plain English"
                           ),
                           tabPanel("Jargon", "a lot of fancy and tech-y words")
                         ),
                         box(
                           title = "Plot",
                           width = 8,
                           status = "success", 
                           solidHeader = TRUE
                         ),
                         box(
                           title = "Example",
                           width = 8 ,
                           position = "right"
                         )
                ),
                tabPanel("SVM",
                         tabBox(
                           width = 4,
                           selected = "Plain English",
                           tabPanel("Plain English", svm_dummies),
                           tabPanel("Jargon", svm_smarties),
                           tabPanel("Example",
                                    "Test data details:",
                                    tableOutput("testData_svm"),
                                    "Test data quality prediction:",
                                    verbatimTextOutput("testQuality_svm"),
                                    "Test data real quality:",
                                    verbatimTextOutput("realQuality_svm"),
                                    actionButton("randomize_svm", "Randomize data!")
                           )
                         ),
                         box(
                           title = "Plot",
                           width = 8,
                           status = "success", 
                           solidHeader = TRUE,
                           radioButtons("vsize_svm", "Select the training sample size:",
                                        c("80/20" = "eightyTwenty",
                                          "64/33" = "oneThird",
                                          "50/50" = "halfHalf"),
                                        inline = TRUE
                           ),
                           checkboxGroupInput("vc_svm", "Select Cost value(s):",
                                              choiceNames = list("2", "4", "6", "8", "10"),
                                              choiceValues = list("2", "4", "6", "8", "10"),
                                              selected = "2", inline = TRUE
                           ),
                           plotOutput("plot_svm")
                         )
                  ),
                  tabPanel("Random Forest",
                         tabBox(
                           width = 4,
                           selected = "Plain English",
                           tabPanel("Plain English"),
                           tabPanel("Jargon", "a lot of fancy and tech-y words"),
                           tabPanel("Example",
                                    print("Test data details:"),
                                    tableOutput("testData_rf"),
                                    print("Test data quality prediction:"),
                                    verbatimTextOutput("testQuality_rf"),
                                    print("Test data real quality:"),
                                    verbatimTextOutput("realQuality_rf"),
                                    actionButton("randomize_rf", "Randomize data!")
                            )
                         ),
                         box(
                           title = "Plot",
                           width = 8,
                           status = "success", 
                           solidHeader = TRUE,
                           radioButtons("vsize_rf", "Select the training sample size:",
                                        c("80/20" = "eightyTwenty",
                                          "64/33" = "oneThird",
                                          "50/50" = "halfHalf"),
                                        inline = TRUE),
                          plotOutput("plot_rf")
                        )
                ),
                tabPanel("Neural Network",
                         tabBox(
                           width = 4,
                           selected = "Plain English",
                           tabPanel("Plain English"),
                           tabPanel("Jargon", "a lot of fancy and tech-y words"),
                           tabPanel("Example",
                                    print("Test data details:"),
                                    tableOutput("testData_nn"),
                                    print("Test data quality prediction:"),
                                    verbatimTextOutput("testQuality_nn"),
                                    print("Test data real quality:"),
                                    verbatimTextOutput("realQuality_nn"),
                                    actionButton("randomize_nn", "Randomize data!"))
                         ),
                         box(
                           title = "Plot",
                           width = 8,
                           status = "success", 
                           solidHeader = TRUE,
                           radioButtons("vsize_nn", "Select the training sample size:",
                                        c("80/20" = "eightyTwenty",
                                          "64/33" = "oneThird",
                                          "50/50" = "halfHalf"),
                                        inline = TRUE
                           ),
                           checkboxGroupInput("vl3_nn", "Select Layer 3 perceptron number(s):",
                                              choiceNames = list("5", "6", "7", "8", "9"),
                                              choiceValues = list("5", "6", "7", "8", "9"),
                                              selected = "5", inline = TRUE
                           ),
                           plotOutput("plot_nn")
                         )
                )
              )
    ),
    
    tabItem(tabName = "compare",
            fluidRow(
              box(
                title = "Model 1", 
                width = 4, 
                status = "success", 
                solidHeader = TRUE
              ),
              box(
                title = "Decision Boundaries", 
                width = 8, 
                status = "primary", 
                solidHeader = TRUE
              ),
              box(
                title = "Model 2",
                width = 4,
                status = "success",
                solidHeader = TRUE
              )
            )
    ),
    
    tabItem(tabName = "game",
            titlePanel("The Winemaker"),
            tags$head(
              tags$style(type="text/css", "
                         label.control-label, 
                         .selectize-control.single{ display: table-cell; text-align: left; vertical-align: middle; } 
                         .form-inline { display: table-row;}
                         ")
              ),
            sidebarLayout(
              sidebarPanel(
                p("Make the best wine that fits the taste of our machine-learning model!
                  Enter the variables below, and click on Test button.
                  Your wine will be judged on a scale of 1 (worst) to 7 (best)."),
                
                column(6, style="padding-top:40px",
                       sliderInput("var01", label="Fixed acidity", 3.5, 12, 15.5/2, step=0.05),
                       sliderInput("var02", label="Volatile acidity", 0, 1, 0.5, step = 0.05),
                       sliderInput("var03", label="Citric acid", 0, 1.8, 0.9, step = 0.05),
                       sliderInput("var04", label="Residual sugar", 0.6, 24, 12, step=0.05),
                       sliderInput("var05", label="Chlorides", 0.01, 0.25, 0.1, step=0.005),
                       sliderInput("var06", label="Free SO2", 3, 131, 134/2, step=0.5)
                ),
                column(6, style="padding-top:40px",
                       sliderInput("var07", label="Total SO2", 18, 315, 333/2,step=0.2),
                       sliderInput("var08", label="Density", 0.98, 1.05, 0.99, step=0.005),
                       sliderInput("var09", label="pH", 2.7, 3.9, 3.3, step=0.01),
                       sliderInput("var10", label="Sulphates", 0.26, 1.05, 0.8,  step=0.05),
                       sliderInput("var11", label="Alcohol", 8, 14.1, 11, step=0.05),
                       actionButton("submit2", "Test", style="padding: 20px; padding-left: 90px; padding-right: 90px; font-size: 200%; horizontal-align: middle")
                )
              ),
              mainPanel(
                h3("Your wine's quality:"),
                verbatimTextOutput("quality"),
                imageOutput("wineglass")
              )
            )
          )
  )
)
    

ui <- dashboardPage(header, sidebar, body)

# ------------------    SERVER    -------------------- #
server <- function(input, output, session){
  
  ############ kNN ################
  vsizeget <- reactive(input$vsize)
  vkernelget <- reactive(input$vkernel)
  
  knETdf = kknnET.train$results[kknnET.train$results$distance == 2,]
  knOTdf = kknnOT.train$results[kknnOT.train$results$distance == 2,]
  knHHdf = kknnHH.train$results[kknnHH.train$results$distance == 2,]
  
  testData <- eightTwoTest.white[1,]
  
  testQuality <- predict(kknnET.train, testData)
  
  observeEvent(input$randomize, {
    testData <- eightTwoTest.white[sample(1:nrow(eightTwoTest.white), 1),]
    
      
    testQuality <- predict(kknnET.train, testData)
    cat("Button pressed!")
    
    output$testData <- renderTable(gather(testData, "Feature", "Value"))
      
    
    output$testQuality <- renderText(testQuality) 
    
    output$realQuality <- renderText(testData$quality)
  } )
  
  output$kknnplot1 <- renderPlot(
    
    if(vsizeget() == "eightyTwenty"){
      testData <- eightTwoTest.white[1,]
      testQuality <- predict(kknnET.train, testData)
      ggplot(data = knETdf, aes(x = kmax, y = Accuracy)) + geom_line(data=subset(knETdf, kernel == vkernelget()[1] |
                                                                                   kernel == vkernelget()[2] |
                                                                                   kernel == vkernelget()[3]), aes(colour = kernel)) + ylim(0.5080, 0.6510)
    }
    else if(vsizeget() == "oneThird"){
      testData <- oneThirdTest.white[1,]
      testQuality <- predict(kknnOT.train, testData)
      ggplot(data = knOTdf, aes(x = kmax, y = Accuracy)) + geom_line(data=subset(knOTdf, kernel == vkernelget()[1] |
                                                                                   kernel == vkernelget()[2] |
                                                                                   kernel == vkernelget()[3]), aes(colour = kernel)) + ylim(0.5080, 0.6510)
    }
    else if(vsizeget() == "halfHalf"){
      testData <- halfHalfTest.white[1,]
      testQuality <- predict(kknnHH.train, testData)
      ggplot(data = knHHdf, aes(x = kmax, y = Accuracy)) + geom_line(data=subset(knHHdf, kernel == vkernelget()[1] |
                                                                                   kernel == vkernelget()[2] |
                                                                                   kernel == vkernelget()[3]), aes(colour = kernel)) + ylim(0.5080, 0.6510)
    }
  )
  
  ########## SVM #################
  
  vsizeget_svm <- reactive(input$vsize_svm)
  vcget_svm <- reactive(input$vc_svm)
  
  svmETdf = svmET.train$results
  svmOTdf = svmOT.train$results
  svmHHdf = svmHH.train$results
  testData_svm <- eightTwoTest.white[1,]
  testQuality_svm <- predict(svmET.train, testData_svm)
  
  observeEvent(input$randomize_svm, {
    testData_svm <- eightTwoTest.white[sample(1:nrow(eightTwoTest.white), 1),]
    testQuality_svm <- predict(svmET.train, testData_svm)
    cat("Button pressed!")
    print(testData_svm)
    print(testQuality_svm)
    
    output$testData_svm <- renderTable(gather(testData_svm, "Feature", "Value"))

    output$testQuality_svm <- renderText(testQuality_svm)

    output$realQuality_svm <- renderText(testData_svm$quality)
  })
  
  output$plot_svm <- renderPlot(
    
    if(vsizeget_svm() == "eightyTwenty"){
      ggplot(data = svmETdf, aes(x = sigma, y = Accuracy, color = C)) + geom_line(data=subset(svmETdf, C == vcget_svm()[1])) + geom_point(data=subset(svmETdf, C == vcget_svm()[1])) + 
        geom_line(data=subset(svmETdf, C == vcget_svm()[2])) + geom_point(data=subset(svmETdf, C == vcget_svm()[2])) + 
        geom_line(data=subset(svmETdf, C == vcget_svm()[3])) + geom_point(data=subset(svmETdf, C == vcget_svm()[3])) + 
        geom_line(data=subset(svmETdf, C == vcget_svm()[4])) + geom_point(data=subset(svmETdf, C == vcget_svm()[4])) + 
        geom_line(data=subset(svmETdf, C == vcget_svm()[5])) + geom_point(data=subset(svmETdf, C == vcget_svm()[5])) + 
        ylim(0.6080, 0.6510)
    }
    else if(vsizeget_svm() == "oneThird"){
      ggplot(data = svmOTdf, aes(x = sigma, y = Accuracy, color = C)) + geom_line(data=subset(svmOTdf, C == vcget_svm()[1])) + geom_point(data=subset(svmOTdf, C == vcget_svm()[1])) + 
        geom_line(data=subset(svmOTdf, C == vcget_svm()[2])) + geom_point(data=subset(svmOTdf, C == vcget_svm()[2])) + 
        geom_line(data=subset(svmOTdf, C == vcget_svm()[3])) + geom_point(data=subset(svmOTdf, C == vcget_svm()[3])) + 
        geom_line(data=subset(svmOTdf, C == vcget_svm()[4])) + geom_point(data=subset(svmOTdf, C == vcget_svm()[4])) + 
        geom_line(data=subset(svmOTdf, C == vcget_svm()[5])) + geom_point(data=subset(svmOTdf, C == vcget_svm()[5])) + 
        ylim(0.5080, 0.6510)
    }
    else if(vsizeget_svm() == "halfHalf"){
      ggplot(data = svmHHdf, aes(x = sigma, y = Accuracy, color = C)) + geom_line(data=subset(svmHHdf, C == vcget_svm()[1])) + geom_point(data=subset(svmHHdf, C == vcget_svm()[1])) + 
        geom_line(data=subset(svmHHdf, C == vcget_svm()[2])) + geom_point(data=subset(svmHHdf, C == vcget_svm()[2])) + 
        geom_line(data=subset(svmHHdf, C == vcget_svm()[3])) + geom_point(data=subset(svmHHdf, C == vcget_svm()[3])) + 
        geom_line(data=subset(svmHHdf, C == vcget_svm()[4])) + geom_point(data=subset(svmHHdf, C == vcget_svm()[4])) + 
        geom_line(data=subset(svmHHdf, C == vcget_svm()[5])) + geom_point(data=subset(svmHHdf, C == vcget_svm()[5])) + 
        ylim(0.5080, 0.6510)
    }
  )

  
  
  ######## NEURAL NETWORK #######
  vsizeget <- reactive(input$vsize_nn)
  vl3get <- reactive(input$vl3_nn)
  
  mlpETdf = mlpET.train$results[mlpET.train$results$layer1 == 5,]
  mlpOTdf = mlpOT.train$results[mlpET.train$results$layer1 == 5,]
  mlpHHdf = mlpHH.train$results[mlpET.train$results$layer1 == 5,]
  testData_nn <- eightTwoTest.white[1,]
  testQuality_nn <- predict(mlpET.train, testData_nn)
  
  observeEvent(input$randomize_nn, {
    testData_nn <- eightTwoTest.white[sample(1:nrow(eightTwoTest.white), 1),]
    testQuality_nn <- predict(mlpET.train, testData_nn)
    cat("Button pressed!")
    print(testData_nn)
    print(testQuality_nn)
    output$testData_nn <- renderTable(gather(testData_nn, "Feature", "Value"))
    output$testQuality_nn <- renderText(testQuality_nn)
    output$realQuality_nn <- renderText(testData_nn$quality)
  })
  
  output$plot_nn <- renderPlot(
    
    if(vsizeget() == "eightyTwenty"){
      ggplot(data = mlpETdf, aes(x = layer2, y = Accuracy, color = layer3)) + geom_line(data=subset(mlpETdf, layer3 == vl3get()[1])) + geom_point(data=subset(mlpETdf, layer3 == vl3get()[1])) + 
        geom_line(data=subset(mlpETdf, layer3 == vl3get()[2])) + geom_point(data=subset(mlpETdf, layer3 == vl3get()[2])) + 
        geom_line(data=subset(mlpETdf, layer3 == vl3get()[3])) + geom_point(data=subset(mlpETdf, layer3 == vl3get()[3])) + 
        geom_line(data=subset(mlpETdf, layer3 == vl3get()[4])) + geom_point(data=subset(mlpETdf, layer3 == vl3get()[4])) + 
        geom_line(data=subset(mlpETdf, layer3 == vl3get()[5])) + geom_point(data=subset(mlpETdf, layer3 == vl3get()[5])) + 
        ylim(0.525, 0.56)
    }
    else if(vsizeget() == "oneThird"){
      ggplot(data = mlpOTdf, aes(x = layer2, y = Accuracy, color = layer3)) + geom_line(data=subset(mlpOTdf, layer3 == vl3get()[1])) + geom_point(data=subset(mlpOTdf, layer3 == vl3get()[1])) + 
        geom_line(data=subset(mlpOTdf, layer3 == vl3get()[2])) + geom_point(data=subset(mlpOTdf, layer3 == vl3get()[2])) + 
        geom_line(data=subset(mlpOTdf, layer3 == vl3get()[3])) + geom_point(data=subset(mlpOTdf, layer3 == vl3get()[3])) + 
        geom_line(data=subset(mlpOTdf, layer3 == vl3get()[4])) + geom_point(data=subset(mlpOTdf, layer3 == vl3get()[4])) + 
        geom_line(data=subset(mlpOTdf, layer3 == vl3get()[5])) + geom_point(data=subset(mlpOTdf, layer3 == vl3get()[5])) + 
        ylim(0.525, 0.56)
    }
    else if(vsizeget() == "halfHalf"){
      ggplot(data = mlpHHdf, aes(x = layer2, y = Accuracy, color = layer3)) + geom_line(data=subset(mlpHHdf, layer3 == vl3get()[1])) + geom_point(data=subset(mlpHHdf, layer3 == vl3get()[1])) + 
        geom_line(data=subset(mlpHHdf, layer3 == vl3get()[2])) + geom_point(data=subset(mlpHHdf, layer3 == vl3get()[2])) + 
        geom_line(data=subset(mlpHHdf, layer3 == vl3get()[3])) + geom_point(data=subset(mlpHHdf, layer3 == vl3get()[3])) + 
        geom_line(data=subset(mlpHHdf, layer3 == vl3get()[4])) + geom_point(data=subset(mlpHHdf, layer3 == vl3get()[4])) + 
        geom_line(data=subset(mlpHHdf, layer3 == vl3get()[5])) + geom_point(data=subset(mlpHHdf, layer3 == vl3get()[5])) + 
        ylim(0.525, 0.56)
    }
  )
  
  ###### RANDOM FOREST ########
  vsizeget_rf <- reactive(input$vsize_rf)
  vkernelget_rf <- reactive(input$vkernel_rf)
  
  rfETdf = rfET.train$results
  rfOTdf = rfOT.train$results
  rfHHdf = rfHH.train$results
  testData_rf <- eightTwoTest.white[1,]
  testQuality_rf <- predict(rfET.train, testData_rf)
  
  observeEvent(input$randomize_rf, {
    testData_rf <- eightTwoTest.white[sample(1:nrow(eightTwoTest.white), 1),]
    testQuality_rf <- predict(rfET.train, testData_rf)
    cat("Button pressed!")
    print(testData_rf)
    print(testQuality_rf)
    output$testData_rf <- renderTable(gather(testData_rf, "Feature", "value"))

    output$testQuality_rf <- renderText(testQuality_rf)

    output$realQuality_rf <- renderText(testData_rf$quality)
  } )
  
  output$plot_rf <- renderPlot(
    
    if(vsizeget_rf() == "eightyTwenty"){
      testData <- eightTwoTest.white[1,]
      testQuality_rf <- predict(rfET.train, testData_rf)
      ggplot(data = rfETdf, aes(x = mtry, y = Accuracy)) + geom_line(data=rfETdf) + ylim(0.60, 0.67)
    }
    else if(vsizeget_rf() == "oneThird"){
      testData_rf <- oneThirdTest.white[1,]
      testQuality_rf <- predict(rfOT.train, testData_rf)
      ggplot(data = rfOTdf, aes(x = mtry, y = Accuracy)) + geom_line(data=rfOTdf) +  ylim(0.6, 0.67)
    }
    else if(vsizeget_rf() == "halfHalf"){
      testData_rf <- halfHalfTest.white[1,]
      testQuality_rf <- predict(rfHH.train, testData_rf)
      ggplot(data = rfHHdf, aes(x = mtry, y = Accuracy)) + geom_line(data=rfHHdf) +  ylim(0.6, 0.67)
    }
  )
  
####### GAME ########
  varnames = c("var01", "var02", "var03", "var04", "var05", "var06", "var07", "var08", "var09", "var10", "var11")
  userData <- eightTwoTest.white[1,]
  userData$quality = 0
  qchar <-"0"
  vargetter <- reactive(c(
    input$var01,
    input$var02,
    input$var03,
    input$var04,
    input$var05,
    input$var06,
    input$var07,
    input$var08,
    input$var09,
    input$var10,
    input$var11)
  )
  
  observeEvent(input$submit2, {
    varList <- vargetter()
    userData[1] <- varList[1]
    for(i in 1:11){
      userData[i] <- varList[i]
    }
    output$quality <- renderText(
      
      print(as.character(predict(rfET.train, userData)))
    )
  
    qchar <- as.character(predict(rfET.train, userData))
    print(qchar)
    
    output$wineglass <- renderImage({
      print(qchar)
      if(qchar == "0"){
        return(list(
          src = "./images/wine_1.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "3"){
        return(list(
          src = "./images/wine_3.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "4"){
        return(list(
          src = "./images/wine_4.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "5"){
        return(list(
          src = "./images/wine_5.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "6"){
        return(list(
          src = "./images/wine_6.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "7"){
        return(list(
          src = "./images/wine_7.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "8"){
        return(list(
          src = "./images/wine_7.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
      else if(qchar == "9"){
        return(list(
          src = "./images/wine_7.png",
          contentType = "image/png",
          alt = NULL
        ))
      }
    }, deleteFile = FALSE)
  }
)
  
}



shinyApp(ui, server)


