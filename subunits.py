#########################################################################################################
########################### Subunits ####################################################################
#########################################################################################################

import numpy as np 
from scipy.special import gamma

def printt(s): 
    """ print and write to log file"""
    print(s)
    with open('shape2sas.log','a') as f:
        f.write('%s\n' %s)

def check_dimension(name,dimensions,n):
    """check if the number of input dimensions for the subunit is correct, else return error message"""
    len_dim = len(dimensions)
    if len_dim != n:
            dim = ' dimension ' if n == 1 else ' dimensions '
            were = ' was ' if len_dim == 1 else ' were '
            printt("\nERROR: subunit " + name + " needs " + str(n) + dim + "(provided after --dimensions or -d), but " + str(len_dim) + were + "given: " + str(dimensions) + "\n")
            exit()

class Sphere:
    aliases = ["sphere","ball","sph"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,1)
        self.R = dimensions[0]

    def getVolume(self) -> float:
        """Returns the volume of a sphere"""
        return (4 / 3) * np.pi * self.R**3
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a sphere"""

        Volume = self.getVolume()
        Volume_max = (2*self.R)**3 # box around sphere.
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where(d < self.R) #save points inside sphere
        x_add,y_add,z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        idx = np.where(d > self.R)
        return idx

class HollowSphere:
    aliases = ["hollowsphere","shell"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)
        self.R,self.r = dimensions

    def getVolume(self):
        """Returns the volume of a hollow sphere"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 4 * np.pi * self.R**2 #surface area of a sphere
        else: 
            return (4 / 3) * np.pi * (self.R**3 - self.r**3)

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a hollow sphere"""
        Volume = self.getVolume()
        if self.r == self.R:
            #The hollow sphere is a shell if r=R
            phi = np.random.uniform(0,2 * np.pi, Npoints)
            costheta = np.random.uniform(-1, 1, Npoints)
            theta = np.arccos(costheta)

            x_add = self.R * np.sin(theta) * np.cos(phi)
            y_add = self.R * np.sin(theta) * np.sin(phi)
            z_add = self.R * np.cos(theta)
            return x_add, y_add, z_add
        Volume_max = (2*self.R)**3
        Vratio = Volume_max/Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R, self.R, N)
        d = np.sqrt(x**2 + y**2 + z**2)

        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a hollow sphere"""
        d = np.sqrt(x_eff**2+y_eff**2+z_eff**2)
        if self.r > self.R:
             self.r, self.R = self.R, self.r
        if self.r == self.R:
            idx = np.where(d != self.R)
            return idx
        else:
            idx = np.where((d > self.R) | (d < self.r))
            return idx

class Cylinder:
    aliases = ["cylinder","disc","rod","cyl"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)     
        self.R,self.l = dimensions

    def getVolume(self):
        """Returns the volume of a cylinder"""

        return np.pi * self.R**2 * self.l
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cylinder"""

        Volume = self.getVolume()
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where(d < self.R)
        x_add,y_add,z_add = x[idx],y[idx],z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cylinder"""

        d = np.sqrt(x_eff**2+y_eff**2)
        idx = np.where((d > self.R) | (abs(z_eff) > self.l / 2))
        return idx

class Ellipsoid:
    aliases = ["ellipsoid","ellips"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.c = dimensions

    def getVolume(self):
        """Returns the volume of an ellipsoid"""
        return (4 / 3) * np.pi * self.a * self.b * self.c

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of an ellipsoid"""
        Volume = self.getVolume()
        Volume_max = 2 * self.a * 2 * self.b * 2 * self.c
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a, self.a, N)
        y = np.random.uniform(-self.b, self.b, N)
        z = np.random.uniform(-self.c, self.c, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """check for points within a ellipsoid"""

        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2 + z_eff**2 / self.c**2
        idx = np.where(d2 > 1)

        return idx

class EllipticalCylinder:
    aliases = ["ellipticalcylinder","ellipticalcyl","ellipticalrod"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.l = dimensions

    def getVolume(self):
        """Returns the volume of an elliptical cylinder"""
        return np.pi * self.a * self.b * self.l

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of an elliptical cylinder"""

        Volume = self.getVolume()
        Volume_max = 2 * self.a * 2 * self.b * self.l
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a, self.a, N)
        y = np.random.uniform(-self.b, self.b, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)

        d2 = x**2 / self.a**2 + y**2 / self.b**2
        idx = np.where(d2 < 1)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add 

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Elliptical cylinder"""
        d2 = x_eff**2 / self.a**2 + y_eff**2 / self.b**2
        idx = np.where((d2 > 1) | (abs(z_eff) > self.l / 2))
        return idx

class Cube:
    aliases = ["cube","dice"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,1)     
        self.a = dimensions[0]

    def getVolume(self):
        """Returns the volume of a cube"""
        return self.a**3

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cube"""

        # Volume = self.getVolume()

        N = Npoints
        x_add = np.random.uniform(-self.a, self.a, N)
        y_add = np.random.uniform(-self.a, self.a, N)
        z_add = np.random.uniform(-self.a, self.a, N)

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cube"""

        idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2) | ((abs(x_eff) <= self.b/2) 
            & (abs(y_eff) <= self.b/2) & (abs(z_eff) <= self.b/2)))
        
        return idx

class HollowCube:
    aliases = ["hollowcube","hollowdice"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)     
        self.a,self.b = dimensions

    def getVolume(self) :
        """Returns the volume of a hollow cube"""
        if self.a < self.b:
            self.a, self.b = self.b, self.a
        if self.a == self.b:
            return 6 * self.a**2 #surface area of a cube
        else: 
            return self.a**3 - self.b**3
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a hollow cube"""

        Volume = self.getVolume()
        
        if self.a == self.b:
            #The hollow cube is a shell if a=b
            d = self.a / 2
            N = int(Npoints / 6)
            one = np.ones(N)
            
            #make each side of the cube at a time
            x_add, y_add, z_add = [], [], []
            for sign in [-1, 1]:
                x_add = np.concatenate((x_add, sign * one * d))
                y_add = np.concatenate((y_add, np.random.uniform(-d, d, N)))
                z_add = np.concatenate((z_add, np.random.uniform(-d, d, N)))
                
                x_add = np.concatenate((x_add, np.random.uniform(-d, d, N)))
                y_add = np.concatenate((y_add, sign * one * d))
                z_add = np.concatenate((z_add, np.random.uniform(-d, d, N)))

                x_add = np.concatenate((x_add, np.random.uniform(-d, d, N)))
                y_add = np.concatenate((y_add, np.random.uniform(-d, d, N)))
                z_add = np.concatenate((z_add, sign * one * d))
            return x_add, y_add, z_add
        
        Volume_max = self.a**3
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)

        x = np.random.uniform(-self.a / 2,self.a / 2, N)
        y = np.random.uniform(-self.a / 2,self.a / 2, N)
        z = np.random.uniform(-self.a / 2,self.a / 2, N)

        d = np.maximum.reduce([abs(x), abs(y), abs(z)])
        idx = np.where(d >= self.b / 2)
        x_add,y_add,z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a hollow cube"""

        if self.a < self.b:
            self.a, self.b = self.b, self.a
        
        if self.a == self.b:
            idx = np.where((abs(x_eff)!=self.a/2) | (abs(y_eff)!=self.a/2) | (abs(z_eff)!=self.a/2))
            return idx
        
        else: 
            idx = np.where((abs(x_eff) >= self.a/2) | (abs(y_eff) >= self.a/2) | 
            (abs(z_eff) >= self.a/2) | ((abs(x_eff) <= self.b/2) 
            & (abs(y_eff) <= self.b/2) & (abs(z_eff) <= self.b/2)))

        return idx

class Cuboid:
    aliases = ["cuboid","brick"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.a,self.b,self.c = dimensions

    def getVolume(self):
        """Returns the volume of a cuboid"""
        return self.a * self.b * self.c
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cuboid"""
        Volume = self.getVolume()
        x_add = np.random.uniform(-self.a, self.a, Npoints)
        y_add = np.random.uniform(-self.b, self.b, Npoints)
        z_add = np.random.uniform(-self.c, self.c, Npoints)
        return x_add, y_add, z_add
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Cuboid"""
        idx = np.where((abs(x_eff) >= self.a/2) 
        | (abs(y_eff) >= self.b/2) | (abs(z_eff) >= self.c/2))
        return idx

class CylinderRing:
    aliases = ["cylinderring","ring","cylring","discring","hollowcylinder","hollowdisc","hollowrod"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.R,self.r,self.l = dimensions
    
    def getVolume(self):
        """Returns the volume of a cylinder ring"""
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            return 2 * np.pi * self.R * self.l #surface area of a cylinder
        else: 
            return np.pi * (self.R**2 - self.r**2) * self.l

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a cylinder ring"""
        Volume = self.getVolume()
        if self.r == self.R:
            #The cylinder ring is a shell
            phi = np.random.uniform(0, 2 * np.pi, Npoints)
            x_add = self.R * np.cos(phi)
            y_add = self.R * np.sin(phi)
            z_add = np.random.uniform(-self.l / 2, self.l / 2, Npoints)
            return x_add, y_add, z_add
        Volume_max = 2 * self.R * 2 * self.R * self.l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.l / 2, self.l / 2, N)
        d = np.sqrt(x**2 + y**2)
        idx = np.where((d < self.R) & (d > self.r))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a cylinder ring"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        if self.r > self.R:
            self.R, self.r = self.r, self.R
        if self.r == self.R:
            idx = np.where((d != self.R) | (abs(z_eff) > self.l / 2))
            return idx
        else: 
            idx = np.where((d > self.R) | (d < self.r) | (abs(z_eff) > self.l / 2))
            return idx

class Ellipsoid_shell:
    aliases = ["ellipsoid_shell","ellips_shell"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,4)     
        self.a,self.b,self.c,self.T = dimensions

        # define outer dimensions ( = semiaxis + half thickness)
        self.a_plus,self.b_plus,self.c_plus = self.a+self.T/2,self.b+self.T/2,self.c+self.T/2

    def getVolume(self):
        return (4 / 3) * np.pi * (self.a_plus * self.b_plus * self.c_plus - (self.a_plus-self.T) * (self.b_plus-self.T) * (self.c_plus-self.T))

    def getPointDistribution(self, Npoints):
        Volume = self.getVolume()
        Volume_max = 2 * self.a_plus * 2 * self.b_plus * 2 * self.c_plus
        Vratio = Volume_max / Volume

        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.a_plus, self.a_plus, N)
        y = np.random.uniform(-self.b_plus, self.b_plus, N)
        z = np.random.uniform(-self.c_plus, self.c_plus, N)

        d2_max = x**2 / self.a_plus**2 + y**2 / self.b_plus**2 + z**2 / self.c_plus**2
        d2_min = x**2 / (self.a_plus-self.T)**2 + y**2 / (self.b_plus-self.T)**2 + z**2 / (self.c_plus-self.T)**2
        idx = np.where((d2_max < 1) & (d2_min > 1))
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within an ellipsoid shell"""
        d2_max = x_eff**2 / self.a_plus**2 + y_eff**2 / self.b_plus**2 + z_eff**2 / self.c_plus**2
        d2_min = x_eff**2 / (self.a_plus-self.T)**2 + y_eff**2 / (self.b_plus-self.T)**2 + z_eff**2 / (self.c_plus-self.T)**2
        idx = np.where((d2_max > 1) | (d2_min < 1))

        return idx
        
class Torus:
    aliases = ["torus","toroid","doughnut"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,2)     
        self.R,self.r = dimensions

    def getVolume(self):
        """Returns the volume of a torus"""

        return 2 * np.pi**2 * self.r**2 * self.R

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a torus"""
        Volume = self.getVolume()
        L = 2 * (self.R + self.r)
        l = 2 * self.r
        Volume_max = L*L*l
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-L/2, L/2, N)
        y = np.random.uniform(-L/2, L/2, N)
        z = np.random.uniform(-l/2, l/ 2, N)
        # equation: (R-sqrt(x**2+y**2))**2 + z**2 = (R-d)**2 + z**2 = r
        d = np.sqrt(x**2 + y**2)
        idx = np.where((self.R-d)**2 + z**2 < self.r**2)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]

        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a torus"""
        d = np.sqrt(x_eff**2 + y_eff**2)
        idx = np.where((self.R-d)**2 + z_eff**2 > self.r**2)
        return idx
        
class Hyperboloid:
    aliases = ["hyperboloid", "hourglass", "coolingtower"]
    
    # https://mathworld.wolfram.com/One-SheetedHyperboloid.html
    # https://www.vcalc.com/wiki/hyperboloid-volume

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,3)     
        self.r,self.c,self.h = dimensions

    def getVolume(self):
        """Returns the volume of a hyperboloid"""
        return np.pi * 2*self.h * self.r**2 * ( 1 + (2*self.h)**2 / ( 12 * self.c**2 ) )

    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a hyperboloid"""
        #Volume = self.getVolume()
        L = 2 * self.h
        R = self.r * np.sqrt( 1 + L**2 / (4 * self.c**2 ) )
        Volume_max = 2*self.h * 2*R * 2*R
        Volume = np.pi * L * ( 2 * self.r**2 + R**2 ) / 3
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-R, R, N)
        y = np.random.uniform(-R, R, N)
        z = np.random.uniform(-self.h, self.h, N)
        idx = np.where(x**2/self.r**2 + y**2/self.r**2 - z**2/self.c**2 < 1.0)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add

    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within a Hyperboloid"""
        idx = np.where(x_eff**2/self.r**2 + y_eff**2/self.r**2 - z_eff**2/self.c**2 > 1.0)
        return idx
    
class Superellipsoid:
    aliases = ["superellipsoid","superellips"]

    def __init__(self, dimensions):
        check_dimension(self.aliases[0],dimensions,4)     
        self.R,self.eps,self.t,self.s = dimensions

    @staticmethod
    def beta(a, b):
        """beta function"""
        return gamma(a) * gamma(b) / gamma(a + b)

    def getVolume(self):
        """Returns the volume of a superellipsoid"""
        return (8 / (3 * self.t * self.s) * self.R**3 * self.eps * 
                self.beta(1 / self.s, 1 / self.s) * self.beta(2 / self.t, 1 / self.t))
    
    def getPointDistribution(self, Npoints):
        """Returns the point distribution of a superellipsoid"""
        Volume = self.getVolume()
        Volume_max = 2 * self.R * self.eps * 2 * self.R * 2 * self.R
        Vratio = Volume_max / Volume
        N = int(Vratio * Npoints)
        x = np.random.uniform(-self.R, self.R, N)
        y = np.random.uniform(-self.R, self.R, N)
        z = np.random.uniform(-self.R * self.eps, self.R * self.eps, N)
        d = ((np.abs(x)**self.s + np.abs(y)**self.s)**(self.t/ self.s) 
            + np.abs(z / self.eps)**self.t)
        idx = np.where(d < np.abs(self.R)**self.t)
        x_add, y_add, z_add = x[idx], y[idx], z[idx]
        return x_add, y_add, z_add
    
    def checkOverlap(self, x_eff, y_eff, z_eff):
        """Check for points within the subunit"""
        d = ((np.abs(x_eff)**self.s + np.abs(y_eff)**self.s)**(self.t / self.s) 
        + np.abs(z_eff / self.eps)**self.t)
        idx = np.where(d >= np.abs(self.R)**self.t)
        return idx
