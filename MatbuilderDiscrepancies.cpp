#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <memory>
#include <limits>
#include "CLI11.hpp"

#include "MatrixSamplerClass.h"
#include "MatrixTools.h"

using namespace std;

class Constraint{
public:
    enum Type {Stratified, Net, PropA, PropAprime, Fake};
    Type type;
    std::vector<int> dimensions;
    bool weak = false;
    double weight;
    int start = -1;
    int end = std::numeric_limits<int>::max();
    int max_unbalance = std::numeric_limits<int>::max();

    string tostring() const;
};

string Constraint::tostring() const {
    string name;
    switch (type) {
        case Net:
            name += "(0-m-s)-net ";
            break;
        case Stratified:
            name += "Stratified ";
            break;
        case PropA:
            name += "PropA ";
            break;
        case PropAprime:
            name += "PropAprime ";
            break;
        case Fake:
            name += "F ";
            break;
    }
    for (int v : dimensions){
        name += to_string(v);
    }
    return name;
}

void parseFile(istream& in, vector<Constraint>& constraints, int& s, int& m, int& b){
    constraints.resize(0);
    string mode;
    string type;
    string line;
    while (getline(in, line)) {
        istringstream sline(line);
        char letter;
        sline >> letter;
        if (letter == '#') {
            continue;
        }
        sline.putback(letter);
        string left_part;
        getline(sline, left_part, '=');
        if (!sline.eof()) {
            if (left_part[0] == 's') {
                sline >> s;
            } else if (left_part[0] == 'm') {
                sline >> m;
            } else if (left_part[0] == 'b' || left_part[0] == 'p') {
                sline >> b;
            }
        } else {
            sline = istringstream(line);
            constraints.emplace_back();
            auto &cons = constraints.back();
            sline >> type;
            bool loop = true;
            do {
                if (type == "from") {
                    sline >> cons.start;
                    sline >> type;
                } else if (type == "to") {
                    sline >> cons.end;
                    sline >> type;
                } else if (type == "weak") {
                    cons.weak = true;
                    sline >> cons.weight;
                    sline >> type;
                } else {
                    loop = false;
                }
            } while (loop);
            if (type == "net") {
                cons.type = Constraint::Type::Net;
                char c;
                sline >> c;
                if (c == 'u'){
                    sline >> cons.max_unbalance;
                } else {
                    sline.putback(c);
                }
            } else if (type == "stratified") {
                cons.type = Constraint::Type::Stratified;
            } else if (type == "propA") {
                cons.type = Constraint::Type::PropA;
            } else if (type == "propAprime") {
                cons.type = Constraint::Type::PropAprime;
            } else {
                cerr << "Error Parsing: Unknown constraint type " << type << endl;
                constraints.pop_back();
            }
            int d;

            while (sline >> d) {
                cons.dimensions.push_back(d);
            }
        }

    }
}

double compute_l2(const std::vector<std::vector<double>>& pointset, std::size_t nbpoints, const std::vector<int>& dims)
{          
    // TODO: Optimize this function. 
    // As it will be called in an increasing number of nbpoints, 
    // it is possible to store previous discrepancies to speed up computations. 

    // Adapated from https://github.com/utk-team/utk/blob/master/src/discrepancy/GeneralizedL2Discrepancy.hpp
    const uint N = nbpoints;
    const uint D = dims.size();        
    long double a, factor_b, factor_c;

    a = pow((4.0/3.0), D);
    factor_b = 2.0/(double)N;
        
    factor_c = 1.0/(double)(N);
    factor_c *= factor_c;
    
    long double sumb = 0.0;
    
    #pragma omp parallel for reduction(+:sumb)
    for(unsigned int i=0; i<N; i++)
    {
        long double prodb = 1.0;
        for(unsigned int j=0; j<D; j++)
        {
            double uij = pointset[i][dims[j]];
            prodb *= ((3.0 - uij*uij)/2.0);
        }
        sumb += prodb;
    }
    
    long double sumc = 0.0;
    
    #pragma omp parallel for reduction(+:sumc)
    for(uint i=0; i<N; i++)
    for(unsigned int iprime=0; iprime<N; iprime++)
    {
        long double prodc = 1.0;
        for(unsigned int j=0; j<D; j++)
        {
            double uij = pointset[i][dims[j]];
            double uiprimej = pointset[iprime][dims[j]];
            double m = uij > uiprimej ? uij : uiprimej;//std::max(uij, uiprimej);
            prodc *= (2.0 - m);
        }
        sumc += prodc;
    }
    
    long double tmp0 = factor_b*sumb;
    long double tmp1 = factor_c*sumc;
    

    return sqrtl(a -tmp0 + tmp1);
}

int main(int argc, char** argv)
{
    CLI::App app{"Discrepancy evaluation for Matbuilder"};

    string constraintFilename;
    app.add_option("-c", constraintFilename, "Constraint file name")->required();
    
    string matrixFilename;
    app.add_option("-m", matrixFilename, "Generated matrices")->required();
    
    string discrepanciesFilename;
    app.add_option("-o", discrepanciesFilename, "Output file for discrepancies (default: output to standard output)");
    
    string pointsetFilename;
    app.add_option("-f", pointsetFilename, "Output pointset file. (default: do not output pointset)");
    
    std::vector<std::size_t> npoints;
    app.add_option("--npoints", npoints, "List of point to compute discrepancy on (default: b^1, b^2, ..., b^m");

    bool fullD = false;
    app.add_flag("--fullD", fullD, "Add discrepancy evaluation on all dimensions (default: false)");


    CLI11_PARSE(app, argc, argv);

    std::ifstream constraintFile(constraintFilename);
    if (!constraintFile.is_open())
    {
        std::cerr << "Error: can not open " << constraintFilename << std::endl;
        return 1;
    }

    std::ifstream matrixFile(matrixFilename);
    if (!matrixFile.is_open())
    {
        std::cerr << "Error: can not open " << matrixFilename << std::endl;
        return 1;
    }

    std::ostream* outDiscrepancy = &std::cout;
    std::ofstream outDiscrepancyFile(discrepanciesFilename);
    if (outDiscrepancyFile.is_open())
    {
        outDiscrepancy = static_cast<std::ostream*>(&outDiscrepancyFile);
    }
    else if(discrepanciesFilename.size() != 0)
    {
        std::cout << "Can not open " << discrepanciesFilename << ", output to standard output." << std::endl;
    }

    std::ofstream outPointsetfile(pointsetFilename);

    int s, m, b;
    std::vector<Constraint> constraints;
    parseFile(constraintFile, constraints, s, m, b);

    // Add "all dimension fake constraint"
    if (fullD)
    {
        constraints.push_back(Constraint());
        constraints.back().type = Constraint::Fake;
        for (std::size_t i = 0; i < s; i++) constraints.back().dimensions.push_back(i); 
    }

    if (npoints.size() == 0)
    {
        npoints.reserve(m);
        std::size_t p = b;
        for (std::size_t i = 1; i < m; i++)
        {
            npoints.push_back(p);
            p *= b;
        }
    }
    std::sort(npoints.begin(), npoints.end());
    const std::size_t max_npoints = *std::max_element(npoints.begin(), npoints.end()); 

    std::vector<std::vector<int> > Cs(s, std::vector<int>(m*m));
    readMatrices(matrixFile, m, s, Cs);

    std::vector<std::vector<double>> pointset(max_npoints, std::vector<double>(s, 0.0));

    // Generate pointset
    #pragma omp parallel for
    for (std::size_t i = 0; i < max_npoints; i++)
    {
        for (std::size_t d = 0; d < s; d++)
        {
            pointset[i][d] = getDouble(Cs[d], m, b, i);
        }
    }

    std::vector<std::vector<double>> discrepancies(npoints.size(), std::vector<double>(constraints.size(), 0.0));
    for (std::size_t i = 0; i < npoints.size(); i++)
    {
        for (std::size_t j = 0; j < constraints.size(); j++)
        {
            discrepancies[i][j] = compute_l2(pointset, npoints[i], constraints[j].dimensions);
        }
    }

    // Write outputs
    if (outPointsetfile.is_open())
    {
        outPointsetfile << std::setprecision(std::numeric_limits<double>::digits10 + 1);
        for (std::size_t i = 0; i < max_npoints; i++)
        {
            for (std::size_t d = 0; d < s - 1; d++)
            {
                outPointsetfile << pointset[i][d] << " ";
            }
            outPointsetfile << pointset[i].back() << "\n";
        }
    }

    // Write header, csv format
    for (std::size_t i = 0; i < constraints.size() - 1; i++)
    {
        *(outDiscrepancy) << constraints[i].tostring() << ",";
    }
    *(outDiscrepancy) << constraints.back().tostring() << "\n";

    *(outDiscrepancy) << std::setprecision(std::numeric_limits<double>::digits10 + 1);

    for (std::size_t i = 0; i < discrepancies.size(); i++)
    {
        for (std::size_t j = 0; j < discrepancies[i].size() - 1; j++)
        {
            *(outDiscrepancy) << discrepancies[i][j] << ",";
        }  
        *(outDiscrepancy) << discrepancies[i].back() << "\n";
    }
}